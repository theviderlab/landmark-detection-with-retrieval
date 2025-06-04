import numpy as np

def cvnet_preprocess(x):
    """
    Preprocesamiento para CVNet.

    Args:
        x: numpy array con forma (1, H, W, 3) en rango [0, 255]

    Returns:
        Normalizado al rango [0, 1] y con estadísticas de ImageNet
    """
    x = x.astype('float32') / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 1, 3))
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 1, 3))
    x = (x - mean) / std
    return x