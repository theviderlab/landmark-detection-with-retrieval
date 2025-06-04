def select_local_filter(filter_method):
    """
    Selecciona y devuelve la función de filtrado para características locales basada en el método especificado.

    Args:
        filter_method (str): Método de filtrado a utilizar (por ejemplo, "cross_matching").
                             Si es None, "none" o una cadena vacía, se devuelve None.

    Returns:
        function or None: Función de filtrado correspondiente o None si no se especifica un método válido.
    """
    if filter_method is None or filter_method.lower() in ["none", ""]:
        return None

    if filter_method.lower() == "cross_matching":
        from image_retrieval.local_features.filtering import cross_matching
        return cross_matching

    # Aquí se pueden agregar otros métodos de filtrado en el futuro.
    return None

def select_local_similarity(local_similarity_config):
    """
    Selecciona y devuelve el objeto de similitud local basado en la configuración.

    Args:
        local_similarity_config (dict): Configuración para la similitud local.
            Ejemplo:
            {
                "method": "similarity_with_bow",
                ... (otros parámetros necesarios)
            }

    Returns:
        Instancia de LocalSimilarityWithBOW o None si el método no coincide.
    """
    method = local_similarity_config.get("method", "default").lower() if local_similarity_config.get("method") else "default"
    if method == "similarity_with_bow":
        from image_retrieval.local_features.similarity_search import LocalSimilarityWithBOW
        return LocalSimilarityWithBOW(config=local_similarity_config)
    else:
        return None


def select_global_similarity(global_similarity_config):
    """
    Selecciona y devuelve el objeto de similitud global basado en la configuración.

    Args:
        global_similarity_config (dict): Configuración para la similitud global.
            Ejemplo:
            {
                "method": "dot",  # o "knn"
                ... (otros parámetros)
            }

    Returns:
        Instancia de una clase de similitud global o None.
    """
    method = global_similarity_config.get("method", "default").lower() if global_similarity_config.get("method") else "default"
    if method == "knn":
        from image_retrieval.global_features.similarity_search import kNNSimilarity
        return kNNSimilarity(config=global_similarity_config)
    elif method == "dot":
        from image_retrieval.global_features.similarity_search import DotProductSimilarity
        return DotProductSimilarity(config=global_similarity_config)
    else:
        return None


