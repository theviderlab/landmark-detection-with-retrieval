from .resnet import ResNet
import torch
import torch.nn as nn
import os

class CVNet(nn.Module):
    def __init__(self, RESNET_DEPTH, REDUCTION_DIM):
        super(CVNet, self).__init__()

        self.encoder_q = ResNet(RESNET_DEPTH, REDUCTION_DIM)

        # Cargar los pesos: checkpoint oficial de CVNet (CVPR2022_CVNet_R50.pyth) o
        # el state_dict ya extraído del encoder (CVNet_50_2048.pth)
        module_dir = os.path.dirname(__file__)
        candidates = [
            os.path.join(module_dir, "models", "CVPR2022_CVNet_R50.pyth"),
            os.path.join(module_dir, "models", "CVNet_50_2048.pth"),
            os.path.join(module_dir, "CVNet_50_2048.pth"),
        ]
        ckpt_path = next((p for p in candidates if os.path.isfile(p)), None)
        if ckpt_path is None:
            raise FileNotFoundError(
                "No se encontró el checkpoint de CVNet. Descarga CVPR2022_CVNet_R50.pyth "
                f"en: {os.path.join(module_dir, 'models')}"
            )

        # Cargar el estado (state_dict) de PyTorch
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        # Checkpoint oficial: {'model_state': {'encoder_q.*': ..., ...}, ...}
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            prefix = "encoder_q."
            state_dict = {
                k[len(prefix):]: v for k, v in checkpoint["model_state"].items() if k.startswith(prefix)
            }
        # Si el checkpoint es un state_dict directamente:
        elif isinstance(checkpoint, dict) and "state_dict" not in checkpoint:
            state_dict = checkpoint
        # Si el checkpoint viene empaquetado como {'state_dict': ..., ...}
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            raise RuntimeError(f"Checkpoint inesperado en {ckpt_path}; no contiene 'state_dict' ni es dict plano.")

        # Cargar los pesos en la ResNet
        self.encoder_q.load_state_dict(state_dict, strict=True)

        self.encoder_q.eval()

    def forward(self, image):
        # compute query features
        return self.encoder_q(image)                               