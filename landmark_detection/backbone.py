from .resnet import ResNet
import torch
import torch.nn as nn
import os

class CVNet(nn.Module):
    def __init__(self, RESNET_DEPTH, REDUCTION_DIM):
        super(CVNet, self).__init__()

        self.encoder_q = ResNet(RESNET_DEPTH, REDUCTION_DIM)

        # Cargar los pesos
        module_dir = os.path.dirname(__file__)
        ckpt_path = os.path.join(module_dir, "CVNet_50_2048.pth")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"No se encontró el checkpoint en: {ckpt_path}")
        
        # Cargar el estado (state_dict) de PyTorch
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        # Si el checkpoint es un state_dict directamente:
        if isinstance(checkpoint, dict) and "state_dict" not in checkpoint:
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