import torch
import torch.nn as nn
import numpy as np
from image_retrieval.backbones.cvnet.cvnet_model import CVNet_Rerank

class CVNetBackbone(nn.Module):
    def __init__(self, depth=50, reduction_dim=256, pretrained_weights=None):
        super().__init__()
        self.model = CVNet_Rerank(depth, reduction_dim)
        if pretrained_weights:
            full_checkpoint = torch.load(pretrained_weights, map_location='cpu')
            if 'model_state' in full_checkpoint:
                state_dict = full_checkpoint['model_state']
            else:
                state_dict = full_checkpoint
            self.model.load_state_dict(state_dict)
        self.model.eval()

    def extract_global(self, x):
        return self.model.extract_global_descriptor(x)

    def extract_local(self, x):
        return self.model.extract_featuremap(x)

    def forward(self, x):
        """
        Acepta un np.ndarray (n, H, W, C) y devuelve el feature map (n, h, w, c).

        Args:
            x (np.ndarray): Entrada (n, H, W, C)

        Returns:
            np.ndarray: Feature map (n, h, w, c)
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).permute(0, 3, 1, 2).float()  # (n, C, H, W)

        with torch.no_grad():
            feat_map = self.model.encoder_q.forward_featuremap(x)  # (n, C, h, w)
            feat_map = feat_map.permute(0, 2, 3, 1)  # (n, h, w, c)
            return feat_map.cpu().numpy()

    def predict(self, x):
        """
        Interfaz compatible con Keras. Equivalente a forward(x).

        Args:
            x (np.ndarray): Entrada (n, H, W, C)

        Returns:
            np.ndarray: Feature map (n, h, w, c)
        """
        return self.forward(x)
