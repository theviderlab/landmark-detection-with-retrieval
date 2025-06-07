import torch.nn as nn
import torch

class PostprocessModule(nn.Module):
    """Redimensiona bounding boxes al tamaño original."""

    def __init__(self, image_dim: tuple[int]):
        super().__init__()
        self.image_dim = torch.tensor(image_dim, dtype=torch.float32)

    def forward(self, boxes: torch.Tensor, orig_size: torch.Tensor):
        scale = torch.stack(
            [
                orig_size[0] / self.image_dim[0],
                orig_size[1] / self.image_dim[1],
                orig_size[0] / self.image_dim[0],
                orig_size[1] / self.image_dim[1],
            ]
        )
        return boxes * scale