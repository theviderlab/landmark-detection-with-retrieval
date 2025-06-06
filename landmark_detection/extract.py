import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms, roi_align
from typing import List
from landmark_detection.backbone import CVNet
from landmark_detection.pooling import SuperGlobalExtractor

class CVNet_SG(nn.Module):
    """
    Versión ONNX‐exportable sin bucles Python pesados:
      - raw_pred: [1, 5 + C, N]
      - image:    [1, 3, H, W]
    Devuelve:
      - scaled_boxes:  Tensor float32 de shape (M+1, K, 4)
      - final_scores:  Tensor float32 de shape (M+1,)
      - final_classes: Tensor int64   de shape (M+1,)
      - crops:         Tensor float32 de shape ((M+1)*K, 3, 224, 224)


    Ahora con filtrado por un conjunto fijo de clases (allowed_classes).
    """

    def __init__(
        self,
        allowed_classes: list[int] = [41,68,70,74,87,95,113,144,150,158,164,165,193,205,212,224,257,
                                      298,310,335,351,354,390,393,401,403,439,442,457,466,489,510,512,
                                      514,524,530,531,543,546,554,565,573,580,587,588,591],
        score_thresh: float = 0.10,
        iou_thresh: float = 0.45,
        scales: List[float] = [0.7071, 1.0, 1.4142],
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float]  = [0.229, 0.224, 0.225],
        rgem_pr: float   = 2.5,
        rgem_size: int   = 5,
        gem_p: float     = 4.6,
        sgem_ps: float   = 10.0,
        sgem_infinity: bool = False,
        eps: float       = 1e-8
    ):
        """
        Args:
          allowed_classes: lista de enteros, e.g. [0, 2, 5, 160]
                            (son los índices de clase que queremos mantener).
          score_thresh:     umbral mínimo de confianza para una detección válida.
          iou_thresh:       umbral de IoU para la NMS.
          scales:           lista de factores de escala para generar versiones ampliadas/reducidas de cada caja.
                            Por defecto [0.7071, 1.0, 1.4142].
        """
        super(CVNet_SG, self).__init__()
        self.score_thresh    = score_thresh
        self.iou_thresh      = iou_thresh
        self.scales          = scales

        # Convertimos allowed_classes a tensor constante, tipo int64.
        self.register_buffer("allowed_cl", torch.tensor(allowed_classes, dtype=torch.int64))

        # Almacenamos mean y std como buffers con shape (1, 3, 1, 1)
        mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std_tensor  = torch.tensor(std,  dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean_cam", mean_tensor)
        self.register_buffer("std_cam",  std_tensor)

        # Instanciamos CVNet
        self.cvnet = CVNet(RESNET_DEPTH=50, REDUCTION_DIM=2048).eval()

        # Instanciamos SuperGlobal
        self.sg = SuperGlobalExtractor(
            rgem_pr = rgem_pr,
            rgem_size = rgem_size,
            gem_p = gem_p,
            sgem_ps = sgem_ps,
            sgem_infinity = sgem_infinity,
            eps = eps
        ).eval()

    def forward(self, raw_pred: torch.Tensor, image: torch.Tensor):
        """
        raw_pred: Tensor(float32) de shape [1, 5 + C, N]
        image:    Tensor(float32) de shape [1, 3, H, W], rangos en [0,1]

        Returns:
        scaled_boxes  -> Tensor float32 de shape (M+1, 4)    en formato (x1, y1, x2, y2)
        final_scores  -> Tensor float32 de shape (M+1,)
        final_classes -> Tensor int64   de shape (M+1,)
        crops         -> Tensor float32 de shape (M·K+1, 3, 224, 224)
        """
        # Decodificar coordenadas y extraer cls_conf + cls_ids
        boxes_all, scores_all, classes_all = self._decode_and_score(raw_pred)

        # Non-Maximum Suppression (NMS) sobre todas las detecciones decodificadas
        keep_nms = nms(boxes_all, scores_all, self.iou_thresh)
        boxes_nms   = boxes_all[keep_nms]    # (R, 4)
        scores_nms  = scores_all[keep_nms]   # (R,)
        classes_nms = classes_all[keep_nms]  # (R,)

        # Filtrar por umbral de score + pertenencia a allowed_classes
        boxes_filt, scores_filt, classes_filt = self._filter_by_score_and_class(
            boxes_nms, scores_nms, classes_nms
        )

        # Si no quedó ninguna caja válida, generar tensores vacíos
        if boxes_filt.numel() == 0:
            device = boxes_all.device
            boxes_filt = torch.zeros((0, 4),  dtype=torch.float32, device=device)
            scores_filt  = torch.zeros((0,),    dtype=torch.float32, device=device)
            classes_filt = torch.zeros((0,),    dtype=torch.int64,   device=device)

        # Añadir siempre la “detección” de la imagen completa al inicio
        _, _, H, W = image.shape
        full_box = torch.tensor(
            [[0.0, 0.0, float(W - 1), float(H - 1)]],
            dtype=boxes_filt.dtype,
            device=boxes_filt.device
        )  # (1, 4)
        full_score = torch.tensor(
            [1.0],
            dtype=scores_filt.dtype,
            device=scores_filt.device
        )  # (1,)
        full_class = torch.tensor(
            [-1],
            dtype=classes_filt.dtype,
            device=classes_filt.device
        )  # (1,)

        final_boxes   = torch.cat([full_box, boxes_filt], dim=0)    # (M+1, 4)
        final_scores  = torch.cat([full_score, scores_filt], dim=0) # (M+1,)
        final_classes = torch.cat([full_class, classes_filt], dim=0)# (M+1,)

        # Escalar todas las cajas según las escalas definidas
        scaled_boxes = self._scale_boxes(final_boxes, image.shape, self.scales)
        # scaled_boxes: shape (M+1, K, 4)

        # Extraer y redimensionar crops para cada (M+1) × K cajas
        crops = self._extract_and_resize_crops(scaled_boxes, image)
        # crops: shape ( (M+1)*K, 3, 224, 224 )

        # Normalizar crops usando función aparte
        crops_norm = self._normalize_crops(crops)

        # Obtener mapas de activación
        with torch.no_grad():
            feature_maps = self.cvnet(crops_norm) 
            descriptors = self.sg(feature_maps, len(self.scales))

        descriptors = F.normalize(descriptors, p=2, dim=1)

        return final_boxes, final_scores, final_classes, descriptors

    def _decode_and_score(self, raw_pred: torch.Tensor):
        """
        Aplica:
          - squeeze(0) y permute(1,0) para pasar de [1, 5+C, N] a [N, 5+C].
          - Convertir cx,cy,w,h → x1,y1,x2,y2
          - Extraer cls_logits (N, C), sacar cls_conf y cls_ids.
        """
        # raw_pred: [1, 5+C, N] → squeeze y permute → [N, 5+C]
        assert raw_pred.shape[0] == 1, "Se espera batch_size == 1 en raw_pred"
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
        cls_logits = x[:, 4:]                         # (N, C)

        # Obtener la confianza y el índice de clase
        cls_conf, cls_ids = cls_logits.max(dim=1)     # ambos (N,)
        cls_ids = cls_ids                         

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
        cls_exp = classes.unsqueeze(1)                # (N, 1)
        allowed_exp = self.allowed_cl.unsqueeze(0)    # (1, K)
        mask_cls = (cls_exp == allowed_exp).any(dim=1)  # (N,)

        # 3) Máscara combinada
        final_mask = mask_score & mask_cls  # ambos (N,)

        # 4) Aplicar máscara a cada tensor
        boxes_filt   = boxes[final_mask]    # (K',4)
        scores_filt  = scores[final_mask]   # (K',)
        classes_filt = classes[final_mask]  # (K',)

        return boxes_filt, scores_filt, classes_filt

    def _scale_boxes(
        self,
        boxes: torch.Tensor,    # shape (M, 4), en formato (x1, y1, x2, y2)
        image_shape: torch.Size, # torch.Size([1, 3, H, W]) o simplemente tuple (H, W)
        scales: List[float]
    ) -> torch.Tensor:
        """
        Dada una lista de M cajas en `boxes` y una lista de factores `scales=[s0, s1, ..., s_{K-1}]`,
        devuelve un tensor scaled_boxes de shape (M, K, 4) donde scaled_boxes[i, j] es
        la caja i original escalada por scales[j], manteniendo el mismo centro.

        - boxes: FloatTensor de shape (M, 4), cada fila (x1, y1, x2, y2).
        Se asume coordenadas absolutas en la imagen (0 <= x < W, 0 <= y < H).
        - image_shape: tamaño de la imagen sobre la que se recortan, puede ser torch.Size([1,3,H,W]) o (H,W).
        - scales: lista de floats con los factores de ampliación (por ejemplo [0.75, 1.0, 1.25]).

        Retorna:
        scaled_boxes: FloatTensor de shape (M, K, 4) en formato (x1', y1', x2', y2').
        """
        # Extraer H,W
        if len(image_shape) == 4:
            _, _, H, W = image_shape
        else:
            H, W = image_shape

        device = boxes.device
        M = boxes.shape[0]
        K = len(scales)

        # Paso 1: calcular centro y tamaño de cada caja original
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        cx = (x1 + x2) * 0.5  # (M,)
        cy = (y1 + y2) * 0.5  # (M,)
        w_orig = x2 - x1      # (M,)
        h_orig = y2 - y1      # (M,)

        # Paso 2: preparar un tensor vacío para todas las K escalas
        # scaled_boxes tendrá shape (M, K, 4)
        scaled_boxes = torch.zeros((M, K, 4), dtype=boxes.dtype, device=device)

        # Paso 3: iterar sobre cada escala y rellenar scaled_boxes[:, j, :]
        for j, s in enumerate(scales):
            w_s = w_orig * s  # (M,)
            h_s = h_orig * s  # (M,)

            # nuevas coordenadas
            x1_s = cx - 0.5 * w_s  # (M,)
            y1_s = cy - 0.5 * h_s  # (M,)
            x2_s = cx + 0.5 * w_s  # (M,)
            y2_s = cy + 0.5 * h_s  # (M,)

            # clamp a los límites de la imagen
            x1_s = x1_s.clamp(min=0.0, max=W - 1.0)
            y1_s = y1_s.clamp(min=0.0, max=H - 1.0)
            x2_s = x2_s.clamp(min=0.0, max=W - 1.0)
            y2_s = y2_s.clamp(min=0.0, max=H - 1.0)

            # almacenar en la “columna” j
            scaled_boxes[:, j, 0] = x1_s
            scaled_boxes[:, j, 1] = y1_s
            scaled_boxes[:, j, 2] = x2_s
            scaled_boxes[:, j, 3] = y2_s

        return scaled_boxes

    def _extract_and_resize_crops(
        self,
        boxes: torch.Tensor,   # (M, K, 4)
        image: torch.Tensor     # (1, 3, H, W)
    ) -> torch.Tensor:
        """
        Dado `boxes` con shape (M, K, 4) —que ya incluye el box de la imagen completa— 
        y `image` en (1,3,H,W), devuelve (M·K, 3, 224, 224) donde:
        • cada una de las M·K filas de `boxes` se recorta de `image` y se redimensiona a 224×224.

        Args:
        - boxes: Tensor float32 de shape (M, K, 4), en formato (x1, y1, x2, y2).
                    Entre esas cajas se encuentra también la de la imagen completa.
        - image: Tensor float32 de shape (1, 3, H, W), rangos en [0,1].

        Returns:
        - all_crops_norm: Tensor float32 de shape (M·K, 3, 224, 224), normalizado con mean/std de ImageNet.
        """
        # M: número de detecciones “originales” (incluyendo full-image), 
        # K: número de escalas por detección
        M, K, _ = boxes.shape
        device = image.device

        # 1) Aplanar las M·K cajas a (M*K, 4)
        boxes_flat = boxes.view(M * K, 4)  # → (M*K, 4)

        # 2) Construir tensor de ROIs (M*K, 5) con batch_index=0
        batch_indices = torch.zeros((M * K, 1), dtype=boxes_flat.dtype, device=device)  # (M*K, 1)
        rois = torch.cat([batch_indices, boxes_flat], dim=1)  # → (M*K, 5)

        # 3) Aplicar RoiAlign de golpe sobre `image` (1,3,H,W)
        crops = roi_align(
            image,           # (1, 3, H, W)
            rois,            # (M*K, 5)
            output_size=(224, 224),
            spatial_scale=1.0,
            sampling_ratio=-1,
            aligned=True
        )  # → (M*K, 3, 224, 224)

        return crops  # (M*K, 3, 224, 224)
    
    def _normalize_crops(self, crops: torch.Tensor) -> torch.Tensor:
        """
        Dado crops de forma (N, 3, H, W), normaliza usando self.mean_cam y self.std_cam.
        """
        # Ambas tienen shape (1, 3, 1, 1), se expanden automáticamente al shape de crops
        return (crops - self.mean_cam) / self.std_cam

