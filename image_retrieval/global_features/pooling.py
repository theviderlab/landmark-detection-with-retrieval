import tensorflow as tf
import numpy as np

class Pooling:
    def __init__(self, method="avg", method_params=None):
        """
        Inicializa la estrategia de pooling.

        Args:
            method (str): 'avg', 'max', 'gem' o 'rgem'.
            method_params (dict): Parámetros específicos para el método elegido.
        """
        self.method = method.lower()
        self.params = method_params if method_params is not None else {}

    def __call__(self, feature_map):
        """
        Aplica el pooling seleccionado sobre un batch de feature maps 4D (n, H, W, C).

        Args:
            feature_map (np.ndarray o tf.Tensor): Feature maps (n, H, W, C) o una sola imagen (H, W, C)

        Returns:
            np.ndarray: (n, C) para 'avg', 'max', 'gem', o (n, H, W, C) para 'rgem'
        """
        if isinstance(feature_map, np.ndarray):
            feature_map = tf.convert_to_tensor(feature_map, dtype=tf.float32)

        if len(feature_map.shape) == 3:
            feature_map = tf.expand_dims(feature_map, axis=0)

        if self.method == "avg":
            pooled = tf.reduce_mean(feature_map, axis=[1, 2])
        elif self.method == "max":
            pooled = tf.reduce_max(feature_map, axis=[1, 2])
        elif self.method == "gem":
            pooled = self.gem_pooling(feature_map, **self.params)
        elif self.method == "rgem":
            pooled = self.rgem_pooling(feature_map, **self.params)
        else:
            raise ValueError(f"Método de pooling '{self.method}' no soportado.")

        return pooled.numpy()

    def gem_pooling(self, x, p=3.0, eps=1e-6):
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

    def rgem_pooling(self, x, pr=2.5, size=5, eps=1e-6):
        """
        Regional Generalized Mean Pooling (Regional-GeM) aplicado de forma vectorizada sobre batches.

        Esta operación corresponde al método Regional-GeM propuesto en el paper:
        Shao, S., Chen, K., Karpur, A., Cui, Q., Araujo, A., Cao, B.:
        "Global features are all you need for image retrieval and reranking".
        In: ICCV (2023)

        Args:
            x (tf.Tensor): Tensor 4D con forma (n, H, W, C)
            pr (float): Potencia para Lp pooling
            size (int): Tamaño del kernel cuadrado
            eps (float): Valor mínimo para evitar inestabilidad numérica

        Returns:
            tf.Tensor: Tensor 4D con forma (n, H, W, C) después de aplicar Regional-GeM
        """
        denom = tf.pow(tf.cast(size ** 2, tf.float32), 1.0 / pr)
        x_norm = x / denom

        pad = (size - 1) // 2
        x_padded = tf.pad(x_norm, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode="REFLECT")

        x_pow = tf.pow(tf.clip_by_value(x_padded, eps, tf.reduce_max(x_padded)), pr)
        pooled = tf.nn.avg_pool2d(x_pow, ksize=size, strides=1, padding="VALID")
        pooled = tf.pow(pooled, 1.0 / pr)

        return 0.5 * pooled + 0.5 * x

    def sgem_fusion(self, descriptors, aug=1, mode="max", p=10.0, eps=1e-8):
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
