from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess

from image_retrieval.backbones.cvnet.cvnet_backbone import CVNetBackbone
from image_retrieval.backbones.cvnet.preprocess import cvnet_preprocess
import os

def select_model(model_config, use_model_preprocess=True):
    """
    Selecciona y construye un modelo backbone convolucional a partir de la configuración proporcionada.

    Este modelo está diseñado específicamente para extracción de características, por lo que:
    - No incluye la top layer (include_top=False)
    - No aplica ningún tipo de pooling final (pooling=None)
    - Devuelve tensores 4D de forma (n, h, w, c), adecuados para aplicar pooling personalizado posteriormente

    Args:
        model_config (dict): Configuración del modelo, con al menos la clave 'model_type'.
        use_model_preprocess (bool): Si es True, se aplica la función de preprocesado correspondiente al modelo.

    Returns:
        model: Modelo backbone de Keras listo para extracción de características.
        preprocess_func: Función de preprocesamiento asociada al modelo.
    """
    model_type = model_config.get("model_type", "resnet50").lower()

    if model_type == "resnet50":
        model = ResNet50(weights='imagenet', include_top=False, pooling=None)
        preprocess_func = resnet_preprocess if use_model_preprocess else lambda x: x
    elif model_type == "vgg16":
        model = VGG16(weights='imagenet', include_top=False, pooling=None)
        preprocess_func = vgg_preprocess if use_model_preprocess else lambda x: x
    elif model_type == "cvnet":
        depth = model_config.get("depth", 50)
        reduction_dim = model_config.get("reduction_dim", 256)
        weights = model_config.get("weights", None)
        # Construir la ruta absoluta del directorio de imágenes
        base_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.normpath(os.path.join(base_dir, weights))

        model = CVNetBackbone(depth=depth, reduction_dim=reduction_dim, pretrained_weights=weights_path)
        preprocess_func = cvnet_preprocess if use_model_preprocess else lambda x: x
    else:
        raise ValueError(f"Modelo {model_type} no soportado.")
    
    return model, preprocess_func