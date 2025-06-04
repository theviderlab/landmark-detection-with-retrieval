import numpy as np
from image_retrieval.global_features.pooling import Pooling
import tensorflow as tf

class FlattenExtractor:
    """
    Extractor de características globales que aplana los feature maps y aplica fusión entre augmentations.

    Compatible con modelos Keras y PyTorch (como CVNetBackbone) que implementen un método .predict()
    que devuelva feature maps con forma (n, h, w, c).
    """

    def __init__(self, model, config=None):
        """
        Args:
            model: Modelo backbone con método .predict(), que devuelve un tensor (n, h, w, c)
            config (dict, optional): Configuración adicional (no utilizada por ahora).
        """
        self.model = model
        self.config = config if config is not None else {}

    def extract(self, preprocessed_img, aug=1):
        """
        Extrae y aplana feature maps, luego fusiona las augmentations por imagen.

        Args:
            preprocessed_img (np.ndarray): Tensor con forma (n, H, W, C), donde n = batch_size * aug
            aug (int): Número de augmentations por imagen. Por defecto 1.

        Returns:
            np.ndarray: Tensor con forma (batch_size, d_global), donde d_global = h * w * c
        """
        # Paso 1: aplicar el modelo → (n, h, w, c)
        feature_maps = self.model.predict(preprocessed_img)

        # Paso 2: flatten → (n, d_global)
        flattened = feature_maps.reshape(feature_maps.shape[0], -1)

        # Paso 3: agrupar por imagen original → (batch_size, aug, d_global)
        n = flattened.shape[0]
        assert n % aug == 0, "La cantidad total de imágenes debe ser divisible por aug"
        batch_size = n // aug
        reshaped = flattened.reshape(batch_size, aug, -1)

        # Paso 4: fusión entre augmentations → (batch_size, d_global)
        final_descriptors = np.mean(reshaped, axis=1)

        return final_descriptors


class SuperGlobalExtractor:
    """
    Extractor global que aplica SuperGlobal Pooling: Regional-GeM (opcional), 
    GeM/Average pooling y fusión entre augmentations (Scale-GeM), 
    según el paper:
    Shao, S., Chen, K., Karpur, A., Cui, Q., Araujo, A., Cao, B.:
    "Global features are all you need for image retrieval and reranking". ICCV (2023)
    """

    def __init__(self, model, config=None):
        """
        Args:
            model: Backbone con método .predict() → (n, h, w, c)
            config (dict): Configuración con claves posibles:
                - 'pooling': str, método principal ('gem', 'avg', etc.)
                - 'method_params': dict, parámetros del método principal
                - 'use_rgem': bool, aplicar Regional-GeM
                - 'rgem_params': dict, parámetros para rgem
                - 'sgem_mode': str, 'max' (SGEM∞) o 'lp' (SGEMp)
                - 'sgem_p': float, potencia p si 'lp'
        """
        self.model = model
        self.config = config if config is not None else {}

        # Regional-GeM
        self.use_rgem = self.config.get("use_rgem", False)
        self.rgem_params = self.config.get("rgem_params", {})
        self.rgem = Pooling(method="rgem", method_params=self.rgem_params) if self.use_rgem else None

        # Pooling principal
        pooling_method = self.config.get("pooling", "gem")
        method_params = self.config.get("method_params", {})
        self.pooling = Pooling(method=pooling_method, method_params=method_params)

        # SGEM (fusión entre augmentations)
        self.sgem_mode = self.config.get("sgem_mode", "lp")  # default: SGEMp
        self.sgem_p = self.config.get("sgem_p", 10.0)

    def extract(self, preprocessed_img, aug=1):
        """
        Extrae descriptores globales siguiendo la arquitectura de SuperGlobal.

        Args:
            preprocessed_img (np.ndarray): Tensor (n, H, W, C), donde n = batch_size * aug
            aug (int): Número de augmentations por imagen

        Returns:
            np.ndarray: Tensor (batch_size, d_global), cada fila es un descriptor global
        """
        n = preprocessed_img.shape[0]
        assert n % aug == 0, "n debe ser divisible por aug"
        batch_size = n // aug

        # 1. Extraer feature maps del backbone
        feature_maps = self.model.predict(preprocessed_img)  # (n, h, w, c)

        # 2. Regional-GeM
        if self.use_rgem:
            feature_maps = self.rgem(feature_maps)  # (n, h, w, c)

        # 3. Pooling (GeM, avg, etc.)
        pooled = self.pooling(feature_maps)  # (n, d_global)

        # 4. Normalización L2 por augmentación
        pooled = tf.linalg.l2_normalize(pooled, axis=1).numpy()  # (n, d_global)

        # 5. SGEM (fusión entre augmentations por imagen)
        final_descriptors = self.pooling.sgem_fusion(
            descriptors=pooled,
            aug=aug,
            mode=self.sgem_mode,
            p=self.sgem_p
        )  # (batch_size, d_global)

        return final_descriptors
