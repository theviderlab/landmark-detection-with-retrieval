

def select_refiner(refine_results_config):
    """
    Selecciona e instancia el refinador de resultados basado en la configuración.

    Args:
        refine_results_config (dict): Diccionario con al menos la clave 'method' y opcionalmente 'params'.

    Returns:
        objeto: Instancia del refinador, o None si no se especifica o se define como 'none'.
    """
    method = refine_results_config.get("method", "none").lower()
    
    if method == "none":
        return None

    elif method == "sg_reranking":
        from image_retrieval.refinement.sg_reranking import SGReranker
        params = refine_results_config.get("params", {})
        return SGReranker(**params)

    else:
        raise NotImplementedError(f"Método de refinamiento '{method}' no está implementado.")
