import torch
import torch.nn.functional as F
import torch.nn as nn

class SuperGlobalExtractor(nn.Module):
    """
    Combina RGEM, GeM y SGEM para procesar mapas de activación de tamaño (N, C, H, W),
    donde N = batch_size * aug. Devuelve descriptores de forma (batch_size, C).
    """
    def __init__(
        self,
        rgem_pr: float   = 2.5,
        rgem_size: int   = 5,
        gem_p: float     = 4.6,
        sgem_ps: float   = 10.0,
        sgem_infinity: bool = False,
        eps: float       = 1e-8
    ):
        super(SuperGlobalExtractor, self).__init__()
        self.rgem = RGEM_Batch(pr=rgem_pr, size=rgem_size)
        self.gem  = GEMp_Batch(p=gem_p, eps=eps)
        self.sgem = SGEM_Batch(ps=sgem_ps, infinity=sgem_infinity, eps=eps)

    def forward(self, feature_maps: torch.Tensor, aug: int) -> torch.Tensor:
        """
        Args:
            feature_maps (torch.Tensor): Tensor con shape (N, C, H, W),
                                         donde N = batch_size * aug.
            aug (int): Número de escalas/augmentations por muestra.

        Returns:
            torch.Tensor: Descriptores globales de forma (batch_size, C).
        """
        # 1) Regional-GeM (N, C, H, W) -> (N, C, H, W)
        x = self.rgem(feature_maps)
        # 2) GeM global (N, C, H, W) -> (N, C)
        x = self.gem(x)
        # 3) Normalización L2 fila a fila (N, C) -> (N, C)
        x = F.normalize(x, p=2, dim=1)
        # 4) SGEM batcheado (N, C) -> (batch_size, C)
        x = self.sgem(x, aug)
        return x

# Ejemplo de uso:
# extractor = SuperGlobalExtractor(
#     rgem_pr=2.5, rgem_size=5, gem_p=4.6, sgem_ps=10.0, sgem_infinity=False, eps=1e-8
# ).eval()
# feature_maps = torch.rand((6, 2048, 7, 7))  # ej. 2 muestras x 3 escalas = 6
# descriptors = extractor(feature_maps, aug=3)  # shape (2, 2048)

class GEMp_Batch(nn.Module):
    """Generalized mean pooling (GeM) adapted for 2D feature maps in batch."""
    def __init__(self, p=4.6, eps=1e-8):
        super(GEMp_Batch, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input feature maps of shape (N, C, H, W).
        Returns:
            torch.Tensor: Pooled descriptors of shape (N, C).
        """
        # Clamp to avoid numerical issues, raise to power p
        x = x.clamp(min=self.eps).pow(self.p)  # (N, C, H, W)
        # Apply adaptive average pooling to get (N, C, 1, 1)
        x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(1, 1))  # (N, C, 1, 1)
        # Take (1/p)-th power and squeeze
        x = x.pow(1.0 / self.p).squeeze(-1).squeeze(-1)  # (N, C)
        return x

# Example usage:
# gem = GEMp_Batch(p=4.6, eps=1e-8)
# input_tensor = torch.rand((8, 2048, 7, 7))
# output = gem(input_tensor)  # shape (8, 2048)

class RGEM_Batch(nn.Module):
    """Regional-GeM idéntico al original (TF/NumPy) para batch de (N, C, H, W)."""
    def __init__(self, pr=2.5, size=5, eps=1e-6):
        super(RGEM_Batch, self).__init__()
        self.pr   = pr
        self.size = size
        self.eps  = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (N, C, H, W)
        Returns:
            torch.Tensor: (N, C, H, W) tras Regional-GeM
        """
        # 1) denom = (size^2)^(1/pr)
        denom = float(self.size ** 2) ** (1.0 / self.pr)

        # 2) x_norm = x / denom
        x_norm = x / denom  # (N, C, H, W)

        # 3) Padding reflect
        pad = (self.size - 1) // 2
        x_padded = torch.nn.functional.pad(x_norm, (pad, pad, pad, pad), mode="reflect")
        # Ahora x_padded es (N, C, H+2pad, W+2pad)

        # 4) x_pow_padded = clamp(x_padded, eps)^pr
        x_pow_padded = torch.clamp(x_padded, min=self.eps).pow(self.pr)  # (N, C, H+2pad, W+2pad)

        # 5) avg_pool2d con kernel=size, stride=1
        pooled = torch.nn.functional.avg_pool2d(x_pow_padded, kernel_size=self.size, stride=1, padding=0)
        # pooled: (N, C, H, W)

        # 6) pooled = clamp(pooled, eps)^(1/pr)
        pooled = torch.clamp(pooled, min=self.eps).pow(1.0 / self.pr)  # (N, C, H, W)

        # 7) resultado final = 0.5 * pooled + 0.5 * x
        return 0.5 * pooled + 0.5 * x

# Example usage:
# rgem = RGEM_Batch(pr=2.5, size=5)
# input_tensor = torch.rand((8, 2048, 7, 7))
# output = rgem(input_tensor)  # shape (8, 2048, 7, 7)

class SGEM_Batch(nn.Module):
    """
    Scale Generalized Mean Pooling (SGEM) batched for (N, d) descriptors.
    N = batch_size * aug
    """
    def __init__(self, ps=10.0, infinity=True, eps=1e-8):
        super(SGEM_Batch, self).__init__()
        self.ps = ps
        self.infinity = infinity
        self.eps = eps

    def forward(self, x: torch.Tensor, aug: int) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input descriptors, shape (N, d) where N = batch_size * aug
            aug (int): Number of augmentations (scales) per sample

        Returns:
            torch.Tensor: Fused descriptors, shape (batch_size, d)
        """
        # x=x.double()
        N, d = x.shape
        if N % aug != 0:
            raise ValueError(f"N={N} no es divisible por aug={aug}")
        B = N // aug  # batch_size

        # Reshape to (B, aug, d)
        reshaped = x.view(B, aug, d)  # (B, aug, d)

        if self.infinity:
            # SGEM∞: normalize each vector (d) and take max over aug
            norms = torch.norm(reshaped, p=2, dim=2, keepdim=True) + self.eps  # (B, aug, 1)
            normalized = reshaped / norms  # (B, aug, d)
            output = normalized.max(dim=1)[0]  # (B, d)
        else:
            # SGEM^p: gamma = minimum per sample over all entries (aug x d)
            gamma = reshaped.view(B, -1).min(dim=1, keepdim=True)[0]  # (B, 1)
            gamma = gamma.view(B, 1, 1)  # (B, 1, 1)
            centered = reshaped - gamma  # (B, aug, d)
            x_pow = centered.pow(self.ps) #centered.clamp(min=self.eps).pow(self.ps)  # (B, aug, d)
            pooled = x_pow.mean(dim=1)  # (B, d)
            output = pooled.pow(1.0 / self.ps) + gamma.view(B, 1)  # (B, d)

        return output

# Example usage:
# sgem = SGEM_Batch(ps=10.0, infinity=False)
# descriptors = torch.rand((6, 2048))  # e.g. 2 samples x 3 scales = 6
# fused = sgem(descriptors, aug=3)  # returns shape (2, 2048)
