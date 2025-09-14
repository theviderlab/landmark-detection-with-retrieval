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
        
        # --- DEBUG: agregar dos cajas extra en coords 640x640 ---
        debug_boxes = torch.tensor(
            [
                [20, 20, 70, 70],          # toda la imagen procesada
            ],
            dtype=final_boxes.dtype,
            device=final_boxes.device,
        )
        final_boxes = torch.cat([final_boxes, debug_boxes], dim=0)

        # agregar scores y clases dummy
        debug_scores = torch.zeros(1, dtype=final_scores.dtype, device=final_scores.device)
        debug_classes = torch.full((1,), -1, dtype=final_classes.dtype, device=final_classes.device)

        final_scores = torch.cat([final_scores, debug_scores], dim=0)
        final_classes = torch.cat([final_classes, debug_classes], dim=0)
        # --- END DEBUG: agregar dos cajas extra en coords 640x640 ---

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
