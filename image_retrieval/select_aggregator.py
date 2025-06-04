def select_feature_aggregator(feature_aggregation_config):
    """
    Selecciona y devuelve el agregador de características según la configuración.

    Args:
        feature_aggregation_config (dict): Configuración para la agregación de características.
            Ejemplo:
            {
                "method": "none"
            }

    Returns:
        aggregator: La instancia del agregador de características. Actualmente, si el método es "none"
                    (o no se especifica ninguno), se retorna None, ya que no se ha implementado ningún método.
                    
    Raises:
        NotImplementedError: Si se especifica un método distinto a "none".
    """
    # Obtener el método definido en la configuración; si no se especifica, se asume "none"
    method = feature_aggregation_config.get("method")
    if not method:
        method = "none"
    else:
        method = method.lower()

    # Si el método es "none", se retorna None (sin agregador implementado)
    if method == "none":
        return None
    else:
        # Para otros métodos, se levanta una excepción ya que aún no se han implementado
        raise NotImplementedError(f"Método de agregación '{method}' no implementado.")
