import numpy as np

def cross_matching(query_feat, refer_feat):
    """
    Realiza la comparación cruzada (cross matching) entre dos imágenes a nivel local y
    devuelve los vectores de características correspondientes a las regiones que tienen
    match mutuo, junto con sus respectivos scores de similitud.

    Se asume que:
      - query_feat y refer_feat tienen forma (M, d_local), donde:
            m es el número de regiones (features locales) y 
            d_local es la dimensión de cada vector.
      - Cada fila (vector) está L2-normalizada.

    Proceso:
      1. Se calcula la matriz de similitud (producto punto) entre cada par de regiones:
             similarity = np.dot(query_feat, refer_feat.T)
         La matriz resultante tendrá forma (M, M).
      2. Para cada región en la imagen de consulta, se obtiene el índice del vector en la imagen de referencia 
         que maximiza la similitud (best_match_for_query). De forma similar, se obtiene, para cada región de refer,
         el índice de la mejor coincidencia en query (best_match_for_refer).
      3. Se valida el match mutuo: para cada región i en query, si la región refer asignada (best_match_for_query[i])
         no tiene a i como su mejor match (es decir, best_match_for_refer[ best_match_for_query[i] ] != i), se descarta
         el match (se asigna 0 al score de i).
      4. Se retornan los vectores de características de query y refer correspondientes a los índices donde el score
         resultante no es cero, junto con los scores.

    Args:
        query_feat (np.array): Array de forma (m, d_local) con las características locales de la imagen de consulta.
        refer_feat (np.array): Array de forma (m, d_local) con las características locales de la imagen de referencia.

    Returns:
        query_selected (np.array): Array de forma (num_nonzero, d_local) con las características locales de las regiones de consulta con match mutuo.
        refer_selected (np.array): Array de forma (num_nonzero, d_local) con las características correspondientes de la imagen de referencia.
        scores (np.array): Array de forma (num_nonzero,) con los scores de similitud para cada región con match mutuo.
                          Si no se encuentra ningún match mutuo, se devuelven arreglos vacíos.
    """
    m = query_feat.shape[0]  # Número de regiones
    # Calcular la matriz de similitud entre cada par de regiones.
    similarity = np.dot(query_feat, refer_feat.T)  # (M, M)
    
    # Para cada región en query, obtener el índice de la región en refer con mayor similitud.
    best_match_for_query = np.argmax(similarity, axis=1)  # (M,)
    # Para cada región en refer, obtener el índice de la región en query con mayor similitud.
    best_match_for_refer = np.argmax(similarity, axis=0)  # (M,)
    
    # Calcular el score máximo para cada región de query.
    scores = np.max(similarity, axis=1)
    
    # Validar el match mutuo: para cada región i, si la región refer asignada no tiene a i como su mejor match, descartar.
    for i in range(m):
        if best_match_for_refer[ best_match_for_query[i] ] != i:
            scores[i] = 0
    
    nonzero = np.where(scores != 0)[0]
    if nonzero.size == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Seleccionar las regiones con match mutuo.
    query_selected = query_feat[nonzero, :]         # Forma: (num_nonzero, d_local)
    # Para cada región seleccionada en query, se toma su mejor match en refer.
    refer_indices = best_match_for_query[nonzero]
    refer_selected = refer_feat[refer_indices, :]      # Forma: (num_nonzero, d_local)
    selected_scores = scores[nonzero]
    
    return query_selected, refer_selected, selected_scores