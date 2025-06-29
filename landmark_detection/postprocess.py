import torch.nn as nn
import torch

class PostprocessModule(nn.Module):
    """Scale detection boxes back to the original image size."""

    def __init__(self, image_dim: tuple[int]):
        super().__init__()
        self.image_dim = torch.tensor(image_dim, dtype=torch.float32)

    def forward(
        self,
        final_boxes: torch.Tensor,
        final_scores: torch.Tensor,
        final_classes: torch.Tensor,
        orig_size: torch.Tensor,
    ):
        scale = torch.stack(
            [
                orig_size[0] / self.image_dim[0],
                orig_size[1] / self.image_dim[1],
                orig_size[0] / self.image_dim[0],
                orig_size[1] / self.image_dim[1],
            ]
        )
        scaled_boxes = final_boxes * scale
        return scaled_boxes, final_scores, final_classes
