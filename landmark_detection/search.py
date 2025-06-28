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
        """
        super(Similarity_Search, self).__init__()

        if not 0.0 <= min_votes <= 1.0:
            raise ValueError("min_votes debe estar entre 0 y 1")

        self.topk = topk
        self.min_sim = min_sim
        self.min_votes = min_votes
        self.remove_inner_boxes = remove_inner_boxes
        self.join_boxes = join_boxes

    def __call__(
        self,
        final_boxes: torch.Tensor | np.ndarray,
        final_scores: torch.Tensor | np.ndarray,
        final_classes: torch.Tensor | np.ndarray,
        descriptors: torch.Tensor | np.ndarray,
        places_db: torch.Tensor | np.ndarray,
    ) -> tuple:
        """Asigna un ``landmark`` a cada detección mediante búsqueda de similitud.

        Parameters
        ----------
        final_boxes : torch.Tensor | numpy.ndarray
            Cajas detectadas por :meth:`Pipeline_Yolo_CVNet_SG.run`.
        final_scores : torch.Tensor | numpy.ndarray
            Confianza de detección (no utilizada).
        final_classes : torch.Tensor | numpy.ndarray
            Clases de detección (no utilizadas).
        descriptors : torch.Tensor | numpy.ndarray
            Descriptores de las detecciones de la consulta.
        places_db : torch.Tensor | numpy.ndarray
            Tensor que concatena los descriptores de la base de datos y los
            ``place_id`` asociados con shape ``(N, C + 1)``.

        Returns
        -------
        tuple
            ``(boxes, sims, classes)`` tras la votación por mayoría.
        """

        Q = descriptors if isinstance(descriptors, torch.Tensor) else torch.tensor(descriptors)
        DB = places_db if isinstance(places_db, torch.Tensor) else torch.tensor(places_db)
        if Q.ndim != 2:
            raise ValueError("descriptors debe tener shape (D, C)")
        if DB.ndim != 2 or DB.shape[1] < 2:
            raise ValueError("places_db debe tener shape (N, C+1)")

        X = DB[:, :-1]
        idx = DB[:, -1].long()

        if Q.shape[1] != X.shape[1]:
            raise ValueError("Dimensión C de Q y X debe coincidir")

        sims = torch.matmul(Q, X.T)  # (D, N)
        top_sims, top_idx = torch.topk(sims, self.topk, dim=1)

        results: list[int | None] = []
        for sim_row, idx_row in zip(top_sims, top_idx):
            mask = sim_row >= self.min_sim
            if not torch.any(mask):
                results.append(None)
                continue
            places = idx[idx_row[mask]]
            unique_ids, counts = torch.unique(places, return_counts=True)
            majority_index = counts.argmax()
            majority = unique_ids[majority_index]
            vote_ratio = counts[majority_index].float() / mask.sum().float()
            if vote_ratio < self.min_votes:
                results.append(None)
                continue
            results.append(int(majority.item()))

        if isinstance(final_boxes, np.ndarray):
            boxes_tensor = torch.from_numpy(final_boxes)
        else:
            boxes_tensor = final_boxes

        if self.remove_inner_boxes is not None and len(results) > 1:
            results = self._remove_overlapping_boxes(boxes_tensor, results, self.remove_inner_boxes)

        # Obtener puntuación de similitud para la clase asignada
        sim_scores: list[float] = []
        for r, sim_row, idx_row in zip(results, top_sims, top_idx):
            if r is None:
                sim_scores.append(0.0)
                continue
            mask = idx[idx_row] == r
            if torch.any(mask):
                sim_scores.append(float(sim_row[mask].max().item()))
            else:
                sim_scores.append(0.0)

        boxes_out = boxes_tensor.detach().cpu().numpy() if isinstance(boxes_tensor, torch.Tensor) else np.asarray(boxes_tensor)
        scores_out = np.asarray(sim_scores, dtype=np.float32)
        classes_out = list(results)

        if self.join_boxes:
            boxes_out, scores_out, classes_out = self._join_boxes_by_class(boxes_out, scores_out, classes_out)

        boxes_out = torch.as_tensor(boxes_out)
        scores_out = torch.as_tensor(scores_out)
        classes_out = torch.as_tensor(
            [-1 if c is None else c for c in classes_out], dtype=torch.int64
        )

        return boxes_out, scores_out, classes_out

    def _remove_overlapping_boxes(
        self,
        boxes: torch.Tensor,
        labels: list[int | None],
        thr: float,
    ) -> list[int | None]:
        """Descarta las cajas grandes cuando se solapan demasiado con otras
        más pequeñas del mismo ``landmark``.

        El solape se calcula como ``intersección / área_menor``.
        """

        b = boxes.detach().cpu().numpy()
        areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        new_labels = list(labels)
        n = len(labels)
        for i in range(n):
            if new_labels[i] is None:
                continue
            for j in range(i + 1, n):
                if new_labels[j] is None or new_labels[i] != new_labels[j]:
                    continue
                x1 = max(b[i, 0], b[j, 0])
                y1 = max(b[i, 1], b[j, 1])
                x2 = min(b[i, 2], b[j, 2])
                y2 = min(b[i, 3], b[j, 3])
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                inter = w * h
                if inter == 0:
                    continue
                if areas[i] > areas[j]:
                    bigger, smaller = i, j
                else:
                    bigger, smaller = j, i
                if inter / areas[smaller] >= thr:
                    new_labels[bigger] = None
        return new_labels

    def _join_boxes_by_class(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: list[int | None],
    ) -> tuple[np.ndarray, np.ndarray, list[int | None]]:
        """Une las cajas de la misma clase en una sola que las englobe."""

        new_boxes: list[list[float]] = []
        new_scores: list[float] = []
        new_labels: list[int | None] = []

        processed: set[int] = set()
        for i, lbl in enumerate(labels):
            if lbl is None or i in processed:
                continue
            idxs = [j for j, l in enumerate(labels) if l == lbl]
            processed.update(idxs)
            cls_boxes = boxes[idxs]
            x1 = float(cls_boxes[:, 0].min())
            y1 = float(cls_boxes[:, 1].min())
            x2 = float(cls_boxes[:, 2].max())
            y2 = float(cls_boxes[:, 3].max())
            new_boxes.append([x1, y1, x2, y2])
            new_scores.append(float(scores[idxs].max()))
            new_labels.append(lbl)

        for i, lbl in enumerate(labels):
            if lbl is None:
                new_boxes.append(boxes[i].tolist())
                new_scores.append(float(scores[i]))
                new_labels.append(None)

        return np.asarray(new_boxes, dtype=boxes.dtype), np.asarray(new_scores, dtype=scores.dtype), new_labels
