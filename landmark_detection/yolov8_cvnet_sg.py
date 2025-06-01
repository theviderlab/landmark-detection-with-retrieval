import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms, roi_align

class PostProcess(nn.Module):
    """
    Versión ONNX‐exportable sin bucles Python pesados:
      - raw_pred: [1, 5 + C, N]
      - image:    [1, 3, H, W]
    Devuelve:
      - boxes_final   (M,4)
      - scores_final  (M,)
      - classes_final (M,)
      - crops         (M,3,224,224)

    Ahora con filtrado por un conjunto fijo de clases (allowed_classes).
    """

    def __init__(
        self,
        allowed_classes: list[int],
        score_thresh: float = 0.10,
        iou_thresh: float = 0.45
    ):
        """
        Args:
          allowed_classes: lista de enteros, e.g. [0, 2, 5, 160]
                            (son los índices de clase que queremos mantener).
          score_thresh:     umbral mínimo de confianza para una detección válida.
          iou_thresh:       umbral de IoU para la NMS.
        """
        super(PostProcess, self).__init__()
        self.score_thresh    = score_thresh
        self.iou_thresh      = iou_thresh

        # Convertimos allowed_classes a tensor constante en CPU, tipo int64.
        # ONNX exportará esta constante en el grafo sin problemas.
        # Corchetes convierten a lista; luego long() asegura dtype int64.
        self.register_buffer("allowed_cl", torch.tensor(allowed_classes, dtype=torch.int64))

    def forward(self, raw_pred: torch.Tensor, image: torch.Tensor):
        """
        raw_pred: Tensor(float32) de shape [1, 5 + C, N]
        image:    Tensor(float32) de shape [1, 3, H, W], rangos en [0,1]

        Returns:
          boxes_final   -> Tensor float32 de shape (M, 4)    en formato (x1, y1, x2, y2)
          scores_final  -> Tensor float32 de shape (M,)
          classes_final -> Tensor int64   de shape (M,)
          crops         -> Tensor float32 de shape (M, 3, 224, 224)
        """
        # 1) Decodificar coordenadas y extraer cls_conf + cls_ids
        boxes_all, scores_all, classes_all = self._decode_and_score(raw_pred)

        # 2) Filtrar por umbral de score + pertenencia a allowed_classes
        boxes_filt, scores_filt, classes_filt = self._filter_by_score_and_class(
            boxes_all, scores_all, classes_all
        )

        # 3) Si no quedó ninguna detección, devolvemos tensores vacíos
        if boxes_filt.numel() == 0:
            return (
                torch.zeros((0, 4),  dtype=torch.float32, device=boxes_filt.device),
                torch.zeros((0,),     dtype=torch.float32, device=boxes_filt.device),
                torch.zeros((0,),     dtype=torch.int64,   device=boxes_filt.device),
                torch.zeros((0, 3, 224, 224), dtype=image.dtype, device=image.device)
            )

        # 4) Non-Maximum Suppression (NMS)
        keep_idx = nms(boxes_filt, scores_filt, self.iou_thresh)  # índices de las detecciones a mantener
        final_boxes   = boxes_filt[keep_idx]       # (M,4)
        final_scores  = scores_filt[keep_idx]      # (M,)
        final_classes = classes_filt[keep_idx]     # (M,)

        # 5) Extraer y redimensionar crops mediante RoiAlign (sin bucle explícito)
        crops = self._extract_and_resize_crops(final_boxes, image)

        return final_boxes, final_scores, final_classes, crops

    def _decode_and_score(self, raw_pred: torch.Tensor):
        """
        Aplica:
          - squeeze(0) y permute(1,0) para pasar de [1, 5+C, N] a [N, 5+C].
          - Convertir cx,cy,w,h → x1,y1,x2,y2
          - Extraer cls_logits (N, C), sacar cls_conf y cls_ids.
        """
        # raw_pred: [1, 5+C, N] → squeeze y permute → [N, 5+C]
        x = raw_pred.squeeze(0).permute(1, 0)  # (N, 5 + C)

        # Coordenadas centrales y tamaño
        cx = x[:, 0]
        cy = x[:, 1]
        w  = x[:, 2]
        h  = x[:, 3]

        # Convertir a boxes (x1, y1, x2, y2)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = torch.stack([x1, y1, x2, y2], dim=1)  # (N,4)

        # Extraer logits de clase: (N, C)
        cls_logits = x[:, 5:]                         # (N, C)

        # Obtener la confianza y el índice de clase (0-based)
        cls_conf, cls_ids = cls_logits.max(dim=1)     # ambos (N,)
        cls_ids = cls_ids + 1                         # pasamos a 1-based, si quieres

        return boxes, cls_conf, cls_ids

    def _filter_by_score_and_class(
        self,
        boxes:   torch.Tensor,   # (N,4)
        scores:  torch.Tensor,   # (N,)
        classes: torch.Tensor    # (N,)
    ):
        """
        Crea una máscara booleana que sea True solamente si:
          - scores > score_thresh
          - classes esté dentro de self.allowed_cl

        Y devuelve los tensores filtrados.
        """
        # 1) Máscara por umbral de score
        mask_score = scores > self.score_thresh    # (N,)

        # 2) Máscara por pertenencia a allowed_classes
        #    Podemos usar torch.isin en versiones modernas de PyTorch,
        #    pero para máxima compatibilidad ONNX haremos un pequeño OR:
        #    (classes == c1) | (classes == c2) | ... para cada c en allowed_cl.
        #    allowed_cl es un buffer de tipo int64 con forma (K,), K = número de clases permitidas.
        #    Vamos a calcular un mask de tamaño (N,) que sea True solo si classes[i] está en allowed_cl.

        # Inicializamos máscara booleana en False para todos
        device = classes.device
        mask_cls = torch.zeros_like(classes, dtype=torch.bool, device=device)  # (N,)

        # Tomamos la lista de allowed_classes (K elementos)
        # y vamos OR-eando: mask_cls |= (classes == allowed_cl[k]) para cada k.
        for c in self.allowed_cl:
            mask_cls = mask_cls | (classes == c)

        # 3) Máscara combinada
        final_mask = mask_score & mask_cls  # ambos (N,)

        # 4) Aplicar máscara a cada tensor
        boxes_filt   = boxes[final_mask]    # (K',4)
        scores_filt  = scores[final_mask]   # (K',)
        classes_filt = classes[final_mask]  # (K',)

        return boxes_filt, scores_filt, classes_filt

    def _extract_and_resize_crops(
        self,
        boxes: torch.Tensor,   # (M, 4)
        image: torch.Tensor    # (1, 3, H, W)
    ) -> torch.Tensor:
        """
        Versión modificada para que primero cree el recorte de toda la imagen,
        luego, si M > 0, obtenga los M recortes de `boxes` con RoiAlign,
        concatene todo, normalice una sola vez y devuelva (M+1, 3, 224, 224).

        Args:
        - boxes: Tensor float32 de shape (M,4) en formato (x1, y1, x2, y2)
        - image: Tensor float32 de shape (1, 3, H, W), valores en [0,1]

        Returns:
        Tensor float32 de shape (M+1, 3, 224, 224):
        • índice 0 = imagen completa escalada a 224×224
        • índices 1..M = recortes de cada caja, en el mismo orden que `boxes`,
            todos normalizados con mean/std de ImageNet.
        """
        M = boxes.shape[0]
        device = image.device

        # 1) Crear el crop de la imagen completa a 224×224
        full_crop = F.interpolate(
            image,
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )  # (1, 3, 224, 224)

        # 2) Si M > 0, obtener recortes mediante RoiAlign
        if M > 0:
            # Formatear ROIs: (M, 5) con [batch_idx, x1, y1, x2, y2]
            batch_indices = torch.zeros((M, 1), dtype=boxes.dtype, device=device)
            rois = torch.cat([batch_indices, boxes], dim=1)  # (M, 5)

            # RoiAlign sobre la imagen original (1,3,H,W)
            crops = roi_align(
                image,            # (1, 3, H, W)
                rois,             # (M, 5)
                output_size=(224, 224),
                spatial_scale=1.0,
                sampling_ratio=-1,
                aligned=True
            )  # (M, 3, 224, 224)

            # Concatenar full_crop (1,3,224,224) con crops (M,3,224,224)
            all_crops = torch.cat([full_crop, crops], dim=0)  # (M+1, 3, 224, 224)
        else:
            # Solo full_crop
            all_crops = full_crop  # (1, 3, 224, 224)

        # 3) Normalizar todos los recortes (M+1, 3, 224, 224) con ImageNet mean/std
        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=device,
            dtype=all_crops.dtype
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=device,
            dtype=all_crops.dtype
        ).view(1, 3, 1, 1)

        all_crops_norm = (all_crops - mean) / std

        return all_crops_norm  # (M+1, 3, 224, 224)
