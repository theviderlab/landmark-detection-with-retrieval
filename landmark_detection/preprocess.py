import torch
import torch.nn as nn
import torch.nn.functional as F

class PreprocessModule(nn.Module):
    """Prepara la imagen para el detector."""

    def __init__(self, image_dim: tuple[int]):
        super().__init__()
        self.image_dim = image_dim

    def forward(self, img_bgr: torch.Tensor):
        h = torch.tensor(img_bgr.shape[0], dtype=torch.float32)
        w = torch.tensor(img_bgr.shape[1], dtype=torch.float32)
        orig_size = torch.stack([w, h])
        img_rgb = img_bgr.permute(2, 0, 1).float()
        img_rgb = img_rgb[[2, 1, 0], ...]  # BGR -> RGB
        img_rgb = img_rgb.unsqueeze(0)
        img_resized = F.interpolate(
            img_rgb,
            size=self.image_dim,
            mode="bilinear",
            align_corners=False,
        )
        img_norm = img_resized / 255.0
        return img_norm, orig_size
