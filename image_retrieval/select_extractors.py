def select_local_extractor(model, local_extractor_config):
    """
    Selecciona y devuelve el extractor local según la configuración.

    Args:
        model: Modelo preentrenado para extraer las características.
        local_extractor_config (dict): Configuración para el extractor local.
            Ejemplo:
            {
                "method": "region_based",
                "num_pool": 200,
                "feat_layer": "block4_conv3",
                "mask_layer": "block5_conv3",
                "filter": "cross_matching"
            }

    Returns:
        extractor: Instancia del extractor local o None si no se especifica un método válido.
    """
    method = local_extractor_config.get("method", "default").lower() if local_extractor_config.get("method") else "default"
    if method == "region_based":
        from image_retrieval.local_features.extraction import RegionBasedExtractor
        return RegionBasedExtractor(model, config=local_extractor_config)
    else:
        # Aquí se pueden agregar otros métodos para extracción local.
        return None

def select_global_extractor(model, global_extractor_config):
    """
    Selecciona y devuelve el extractor global según la configuración.

    Args:
        model: Modelo preentrenado para extraer las características.
        global_extractor_config (dict): Configuración para el extractor global.

    Returns:
        extractor: Instancia del extractor global o None si no se especifica un método.
    """
    method = global_extractor_config.get("method")
    if not method:
        return None

    method = method.lower()

    if method == "predict_flatten":
        from image_retrieval.global_features.extraction import FlattenExtractor
        return FlattenExtractor(model, config=global_extractor_config)

    elif method == "super_global":
        from image_retrieval.global_features.extraction import SuperGlobalExtractor
        return SuperGlobalExtractor(model, config=global_extractor_config)

    else:
        raise ValueError(f"Método de extractor global '{method}' no reconocido.")

