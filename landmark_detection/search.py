import numpy as np
import torch
import torch.nn as nn


class Similarity_Search(nn.Module):
    """Realiza búsqueda de similitud y votación por mayoría para cada detección."""

    def __init__(
        self,
        topk: int = 5,
        min_sim: float = 0.8,
        min_votes: float = 0.0,
        remove_inner_boxes: float | None = None,
        join_boxes: bool = False,
    ) -> None:
        """Inicializa el buscador.

        Parameters
        ----------
        topk : int
            Número máximo de vecinos a considerar.
        min_sim : float
            Similitud mínima para aceptar un vecino.
        min_votes : float, optional
            Porcentaje mínimo de votos necesarios para asignar una clase.
        remove_inner_boxes : float, optional
            Umbral para descartar cajas más grandes que se solapan con otras
            más pequeñas del mismo ``landmark``. Si ``None`` no se aplica
            este filtrado.
        join_boxes : bool, optional
        """
        super(Similarity_Search, self).__init__()

        if not 0.0 <= min_votes <= 1.0:
            raise ValueError("min_votes debe estar entre 0 y 1")

        self.topk = topk
        self.min_sim = min_sim
        self.min_votes = min_votes
        self.remove_inner_boxes = remove_inner_boxes
        self.join_boxes = join_boxes

    def _prepare_descriptors(
        self,
        query_descriptors: torch.Tensor | np.ndarray,
        db_descriptors: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convierte las entradas a tensores y comprueba las dimensiones."""

        Q = (
            query_descriptors
            if isinstance(query_descriptors, torch.Tensor)
            else torch.tensor(query_descriptors)
        )
        X = (
            db_descriptors
            if isinstance(db_descriptors, torch.Tensor)
            else torch.tensor(db_descriptors)
        )

        if Q.ndim != 2 or X.ndim != 2:
            raise ValueError("Los descriptores deben ser tensores 2D")
        if Q.shape[1] != X.shape[1]:
            raise ValueError("Dimensión C de Q y X debe coincidir")

        return Q, X

    def _similarity_matrix(self, query: torch.Tensor, db: torch.Tensor) -> torch.Tensor:
        """Devuelve la matriz de similitud ``(Q, N)``."""

        return torch.matmul(query, db.T)

    def compute_ranks(
        self,
        query_descriptors: torch.Tensor | np.ndarray,
        db_descriptors: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        """Devuelve el ranking de similitud para cada consulta."""

        Q, X = self._prepare_descriptors(query_descriptors, db_descriptors)
        sims = self._similarity_matrix(Q, X)  # (Q, N)
        return torch.argsort(sims, dim=1, descending=True).T

    def forward(
        self,
        boxes: torch.Tensor | np.ndarray,
        descriptors: torch.Tensor | np.ndarray,
        places_db: torch.Tensor | np.ndarray,
    ) -> tuple:
        """Asigna un ``landmark`` a cada detección mediante búsqueda de similitud.

        Parameters
        ----------
        boxes : torch.Tensor | numpy.ndarray
            Cajas detectadas por :meth:`Pipeline_Landmark_Detection.run`.
        descriptors : torch.Tensor | numpy.ndarray
            Descriptores de las detecciones de la consulta.
        places_db : torch.Tensor | numpy.ndarray
            Tensor que concatena los descriptores de la base de datos y los
            ``place_id`` asociados con shape ``(N, C + 1)``.

        Returns
        -------
        tuple
            ``(boxes, sims, classes)`` tras la votación por mayoría.
            Solo se devuelven las detecciones con una clase asignada.
        """

        DB = (
            places_db
            if isinstance(places_db, torch.Tensor)
            else torch.tensor(places_db)
        )
        if DB.ndim != 2 or DB.shape[1] < 2:
            print(DB)  # debug
            raise ValueError("places_db debe tener shape (N, C+1)")

        X = DB[:, :-1]
        idx = DB[:, -1].long()

        Q, X = self._prepare_descriptors(descriptors, X)
        sims = self._similarity_matrix(Q, X)  # (D, N)
        top_sims, top_idx = torch.topk(sims, self.topk, dim=1)

        # IDs de los vecinos en topk para cada detección
        top_ids = idx[top_idx]
        mask_sim = top_sims >= self.min_sim  # (D, K)

        # Conteo de votos por clase usando one-hot
        num_classes = idx.amax() + 1
        classes_range = torch.arange(num_classes, device=idx.device)
        one_hot_ids = (top_ids.unsqueeze(-1) == classes_range).to(torch.float32)
        vote_counts = (one_hot_ids * mask_sim.unsqueeze(-1)).sum(dim=1)

        majority_counts, majority_ids = vote_counts.max(dim=1)
        valid_votes = mask_sim.sum(dim=1).to(dtype=torch.float32)
        vote_ratio = torch.where(
            valid_votes > 0,
            majority_counts / valid_votes,
            torch.zeros_like(valid_votes),
        )

        valid = (majority_counts > 0) & (vote_ratio >= self.min_votes)
        results = torch.where(valid, majority_ids, torch.full_like(majority_ids, -1))

        if isinstance(boxes, np.ndarray):
            boxes_tensor = torch.from_numpy(boxes)
        else:
            boxes_tensor = boxes

        if self.remove_inner_boxes is not None and boxes_tensor.size(0) > 1:
            results = self._remove_overlapping_boxes(
                boxes_tensor, results, self.remove_inner_boxes
            )

        match = (top_ids == results.unsqueeze(1)) & mask_sim
        sim_scores = (
            torch.where(match, top_sims, torch.zeros_like(top_sims)).max(dim=1).values
        )

        boxes_out = boxes_tensor
        scores_out = sim_scores
        classes_out = results

        if self.join_boxes:
            boxes_out, scores_out, classes_out = self._join_boxes_by_class(
                boxes_out, scores_out, classes_out
            )

        valid_mask = (
            (classes_out >= 0)
            & torch.isfinite(scores_out)
            & torch.isfinite(boxes_out).all(dim=1)
        )

        if torch.any(valid_mask):
            idx = valid_mask.nonzero(as_tuple=True)[0]
            boxes_out = boxes_out.index_select(0, idx)
            scores_out = scores_out.index_select(0, idx)
            classes_out = classes_out.index_select(0, idx)
        else:
            boxes_out = boxes_out[:0]
            scores_out = scores_out[:0]
            classes_out = classes_out[:0]

        return boxes_out, scores_out, classes_out

    def _remove_overlapping_boxes(
        self,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        thr: float,
    ) -> torch.Tensor:
        """Descarta cajas grandes que se solapan demasiado con otras
        más pequeñas del mismo ``landmark``.

        El solape se calcula como ``intersección / área_menor``.
        """

        n = boxes.size(0)
        if n == 0:
            return labels

        boxes_f = boxes.to(dtype=torch.float32)
        areas = (boxes_f[:, 2] - boxes_f[:, 0]) * (boxes_f[:, 3] - boxes_f[:, 1])

        x1 = boxes_f[:, 0]
        y1 = boxes_f[:, 1]
        x2 = boxes_f[:, 2]
        y2 = boxes_f[:, 3]

        xi1 = torch.maximum(x1[:, None], x1[None, :])
        yi1 = torch.maximum(y1[:, None], y1[None, :])
        xi2 = torch.minimum(x2[:, None], x2[None, :])
        yi2 = torch.minimum(y2[:, None], y2[None, :])

        inter_w = torch.clamp(xi2 - xi1, min=0)
        inter_h = torch.clamp(yi2 - yi1, min=0)
        inter = inter_w * inter_h

        area_i = areas[:, None]
        area_j = areas[None, :]
        area_small = torch.minimum(area_i, area_j)
        overlap = torch.where(
            area_small > 0, inter / area_small, torch.zeros_like(inter)
        )

        same_lbl = (labels[:, None] == labels[None, :]) & (labels[:, None] >= 0)
        mask_pair = same_lbl & (overlap >= thr)

        idx_range = torch.arange(n, device=boxes.device)
        non_diag = idx_range[:, None] != idx_range[None, :]
        mask_pair = mask_pair & non_diag

        bigger_i = area_i > area_j
        remove_i = (mask_pair & bigger_i).any(dim=1)
        remove_j = (mask_pair & ~bigger_i).any(dim=0)
        remove = remove_i | remove_j

        return torch.where(remove, torch.full_like(labels, -1), labels)

    def _join_boxes_by_class(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Une las cajas de la misma clase en una sola que las englobe."""

        valid = labels >= 0
        if not torch.any(valid):
            return boxes, scores, labels

        lbls = labels[valid]
        unique_lbls, idx = torch.unique(lbls, sorted=True, return_inverse=True)

        min_x1 = torch.full_like(unique_lbls, float("inf"), dtype=boxes.dtype)
        min_y1 = torch.full_like(unique_lbls, float("inf"), dtype=boxes.dtype)
        max_x2 = torch.full_like(unique_lbls, float("-inf"), dtype=boxes.dtype)
        max_y2 = torch.full_like(unique_lbls, float("-inf"), dtype=boxes.dtype)
        max_sc = torch.full_like(unique_lbls, float("-inf"), dtype=scores.dtype)

        min_x1.scatter_reduce_(
            0, idx, boxes[valid, 0], reduce="amin", include_self=True
        )
        min_y1.scatter_reduce_(
            0, idx, boxes[valid, 1], reduce="amin", include_self=True
        )
        max_x2.scatter_reduce_(
            0, idx, boxes[valid, 2], reduce="amax", include_self=True
        )
        max_y2.scatter_reduce_(
            0, idx, boxes[valid, 3], reduce="amax", include_self=True
        )
        max_sc.scatter_reduce_(0, idx, scores[valid], reduce="amax", include_self=True)

        new_boxes = torch.stack([min_x1, min_y1, max_x2, max_y2], dim=1)
        new_scores = max_sc
        new_labels = unique_lbls

        none_boxes = boxes[~valid]
        none_scores = scores[~valid]
        none_labels = labels[~valid]

        if none_boxes.numel() > 0:
            new_boxes = torch.cat([new_boxes, none_boxes], dim=0)
            new_scores = torch.cat([new_scores, none_scores], dim=0)
            new_labels = torch.cat([new_labels, none_labels], dim=0)

        return new_boxes, new_scores, new_labels
