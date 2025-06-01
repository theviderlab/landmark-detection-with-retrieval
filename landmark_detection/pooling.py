import torch
import torch.nn.functional as F

class SuperGlobalExtractor:
    """
    Extractor global que aplica SuperGlobal Pooling: Regional-GeM (opcional), 
    GeM/Average pooling y fusión entre augmentations (Scale-GeM), 
    según el paper:
    Shao, S., Chen, K., Karpur, A., Cui, Q., Araujo, A., Cao, B.:
    "Global features are all you need for image retrieval and reranking". ICCV (2023)
    """

    def __init__(self, gem_p=3.0, rgem_pr=2.5, rgem_size=5, sgem_mode=0, sgem_p=10):

        # Regional-GeM
        self.rgem_pr = rgem_pr
        self.rgem_size = rgem_size

        # GeM
        self.gem_p = gem_p

        # Scale-GEM
        self.sgem_mode = sgem_mode # 1: 'max' para SGEM∞, 0: 'lp' para SGEM^p
        self.sgem_p = sgem_p

        self.eps = 1e-6

    def forward(self, x, aug=1):
        """
        Extrae descriptores globales siguiendo la arquitectura de SuperGlobal.

        Args:
            preprocessed_img (np.ndarray): Tensor (n, H, W, C), donde n = batch_size * aug
            aug (int): Número de augmentations por imagen

        Returns:
            np.ndarray: Tensor (batch_size, d_global), cada fila es un descriptor global
        """

        # 1. Regional-GeM
        x = self.rgem(x, pr=self.rgem_pr, size=self.rgem_size, eps=self.eps)  # (n, h, w, c)

        # 3. Pooling GeM
        pooled = self.gem(self, x, p=self.gem_p, eps=self.eps)  # (n, d_global)

        # 4. Normalización L2 por augmentación
        pooled = tf.linalg.l2_normalize(pooled, axis=1).numpy()  # (n, d_global)

        # 5. SGEM (fusión entre augmentations por imagen)
        final_descriptors = self.sgem(
            descriptors=pooled,
            aug=aug,
            mode=self.sgem_mode,
            p=self.sgem_p
        )  # (batch_size, d_global)

        return final_descriptors

    def gem(self, x, p=3.0, eps=1e-6):
        """
        Generalized Mean Pooling (GeM)

        Args:
            x (tf.Tensor): (n, H, W, C)
            p (float): Potencia para pooling
            eps (float): Estabilidad numérica

        Returns:
            tf.Tensor: (n, C)
        """
        x = tf.clip_by_value(x, eps, tf.reduce_max(x))
        x = tf.pow(x, p)
        x = tf.reduce_mean(x, axis=[1, 2])
        return tf.pow(x, 1.0 / p)

    def rgem(x: torch.Tensor, pr: float = 2.5, size: int = 5, eps: float = 1e-6) -> torch.Tensor:
        """
        Regional Generalized Mean Pooling (Regional‐GeM) en PyTorch.

        Args:
            x (torch.Tensor): Tensor 4D con forma (N, C, H, W)
            pr (float): Potencia para el Lₚ‐pooling (p = pr)
            size (int): Tamaño del kernel cuadrado para el pooling (debe ser impar)
            eps (float): Evita inestabilidad numérica en valores muy pequeños

        Returns:
            torch.Tensor: Tensor 4D con forma (N, C, H, W) tras aplicar Regional‐GeM
        """
        # 1) Escalar x por el factor (size²)^(1/pr)
        denom = float(size * size) ** (1.0 / pr)
        x_norm = x / denom  # (N, C, H, W)

        # 2) Padding reflectivo en H y W
        pad = (size - 1) // 2
        # F.pad en PyTorch para 4D: pad = (pad_left, pad_right, pad_top, pad_bottom)
        x_padded = F.pad(x_norm, (pad, pad, pad, pad), mode="reflect")  # (N, C, H+2pad, W+2pad)

        # 3) Elevar a la potencia pr, pero evitando valores < eps
        #    (se utiliza el máximo global para definir el tope del clamp)
        max_val = x_padded.max()
        x_clamped = torch.clamp(x_padded, min=eps, max=max_val)
        x_pow = x_clamped.pow(pr)  # (N, C, H+2pad, W+2pad)

        # 4) Promedio local con stride=1 y padding=“VALID” (ya hemos aplicado padding manual)
        pooled = F.avg_pool2d(x_pow, kernel_size=size, stride=1, padding=0)  # (N, C, H, W)

        # 5) Elevar resultado a la 1/pr
        pooled = pooled.pow(1.0 / pr)  # (N, C, H, W)

        # 6) Combinar 50% pooled + 50% entrada original
        out = 0.5 * pooled + 0.5 * x  # (N, C, H, W)
        return out

    def sgem(self, descriptors, aug=1, mode="max", p=10.0, eps=1e-8):
        """
        Scale Generalized Mean Pooling (Scale-GeM) para fusionar descriptores globales de distintas escalas.

        Esta operación corresponde al método Scale-GeM (SGEM) propuesto en el paper:
        Shao, S., Chen, K., Karpur, A., Cui, Q., Araujo, A., Cao, B.:
        "Global features are all you need for image retrieval and reranking".
        In: ICCV (2023)

        Args:
            descriptors (np.ndarray): Tensor con forma (n, d_global), donde n = batch_size * aug
            aug (int): Cantidad de augmentations por imagen (escalas)
            mode (str): 'max' para SGEM∞ o 'lp' para SGEM^p
            p (float): Potencia para SGEM^p
            eps (float): Estabilidad numérica

        Returns:
            np.ndarray: Tensor con forma (batch_size, d_global) con descriptores fusionados
        """
        assert descriptors.ndim == 2, "descriptors debe tener forma (n, d_global)"
        n, d = descriptors.shape
        assert n % aug == 0, "n debe ser divisible por aug"
        batch_size = n // aug

        # Reorganizar: (batch_size, aug, d_global)
        reshaped = descriptors.reshape(batch_size, aug, d)

        if mode == "max":
            # L2-normalizar cada vector antes de hacer max
            norms = np.linalg.norm(reshaped, axis=2, keepdims=True) + eps
            normalized = reshaped / norms
            return np.max(normalized, axis=1)  # (batch_size, d_global)

        elif mode == "lp":
            gamma = np.min(reshaped)
            centered = reshaped - gamma
            pooled = np.mean(np.power(centered, p), axis=1)
            return np.power(pooled, 1.0 / p) + gamma

        else:
            raise ValueError(f"Modo SGEM '{mode}' no soportado. Usa 'max' o 'lp'.")