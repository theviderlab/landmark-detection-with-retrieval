from scipy.spatial.distance import cdist
import numpy as np
import pickle
from image_retrieval.select_similarity import select_local_filter
import os 

import numpy as np
import pickle
from scipy.spatial.distance import cdist

class LocalSimilarityWithBOW:
    """
    Calcula la similitud global entre dos imágenes a nivel local utilizando el método Bag-of-Words (BoW).

    Se asume que:
      - Tanto query_feat como refer_feat tienen forma (M, d_local), donde cada fila representa un vector pooled,
        y m puede variar entre imágenes.
      - Se utiliza un vocabulario (cargado desde un archivo pickle) que contiene:
            - "kmeans": modelo k-means entrenado con centros de forma (num_clusters, d_local).
            - "idf": vector de pesos idf de tamaño (num_clusters,).
      - Opcionalmente, se puede aplicar un filtro (por ejemplo, cross_matching) antes de calcular la similitud.
    """

    def __init__(self, config=None):
        """
        Inicializa el comparador cargando la configuración extraída del YAML.

        La configuración (config) debe contener:
            - "vocabulary_file": Ruta al archivo pickle que contiene el vocabulario.
            - "n_neighbors": (opcional) Número de vecinos a devolver en la búsqueda (por defecto 5).
            - "filter": (opcional) Método de filtrado a aplicar en la búsqueda local (por ejemplo, "cross_matching").
        """
        if config is None:
            config = {}
        # Cargar vocabulario desde la ruta indicada.
        vocab_path = config.get("vocabulary_file", None)
        self.n_neighbors = config.get("n_neighbors", 5)
        if vocab_path is None:
            raise ValueError("El vocabulario es obligatorio; se debe especificar 'vocabulary_file' en la configuración.")
        try:
            abs_vocab_path = self._get_abs_path(vocab_path)
            with open(abs_vocab_path, 'rb') as f:
                self.vocabulary = pickle.load(f)
        except Exception as e:
            raise ValueError(f"No se pudo cargar el vocabulario desde '{abs_vocab_path}': {e}")
        
        # Seleccionar el filtro de características locales (la función select_local_filter se asume importada).
        filter_method = config.get("filter", None)
        self.filter_func = select_local_filter(filter_method)

    def similarity(self, query_feat, refer_feat, scores=None):
        """
        Calcula el puntaje global de similitud entre dos imágenes usando características locales y BoW.

        Se asume que:
          - query_feat y refer_feat son arrays de forma (m, d_local), donde cada fila es un vector.
          - Si no se proporcionan scores, se calcula el máximo del producto punto entre cada vector de consulta
            y todos los vectores de referencia, para obtener un score por cada vector de consulta.
          - Luego se reponderan cada uno de estos scores multiplicándolos por el idf correspondiente al centro
            asignado al vector de consulta.
          - La similitud final es el promedio de los scores reponderados.

        Args:
            query_feat (np.array): Array de forma (m_query, d_local) con las características locales de la imagen de consulta.
            refer_feat (np.array): Array de forma (m_ref, d_local) con las características locales de la imagen de referencia.
            scores (np.array, opcional): Vector de scores para cada vector de consulta, de longitud m_query.

        Returns:
            score (float): Puntaje global de similitud.
        """
        m = query_feat.shape[0]
        if m == 0:
            raise ValueError("La cantidad de regiones (m) es 0.")
        
        # Si scores no se proporcionan, calcular el máximo de la similitud para cada vector de consulta.
        # Se usa el producto punto: (m_query, d) dot (d, m_ref) = (m_query, m_ref)
        if scores is None:
            similarity_matrix = np.dot(query_feat, refer_feat.T)
            scores = np.max(similarity_matrix, axis=1)
        
        # Obtener centros e idf del vocabulario.
        C = self.vocabulary["kmeans"].cluster_centers_  # (num_clusters, d_local)
        idf = self.vocabulary["idf"]
        
        # Calcular la asignación de cada vector de consulta al centro más cercano.
        # Usamos cdist: query_feat de forma (m_query, d_local) y C de forma (num_clusters, d_local)
        query_dist = cdist(query_feat, C, metric='euclidean')  # (m_query, num_clusters)
        q_index = np.argmin(query_dist, axis=1)  # Para cada vector de consulta, el índice del centro más cercano.
        
        # Ponderar cada score con el idf correspondiente.
        for j in range(m):
            scores[j] *= idf[q_index[j]]
        
        final_score = np.sum(scores) / m
        return final_score

    def search(self, query_feat, db_feat, n_neighbors=None):
        """
        Realiza la búsqueda de las imágenes más similares en la base de datos utilizando el método similarity.
        Si se ha configurado un filtro, se aplica antes de calcular la similitud para cada imagen.

        Args:
            query_feat (np.array): Array de características locales de la imagen de consulta, forma (m_query, d_local).
            db_feat (iterable): Lista de arrays; cada array es de forma (m_i, d_local) para una imagen de la base de datos.
            n_neighbors (int, opcional): Número de imágenes similares a devolver. Si no se especifica, se utiliza self.n_neighbors.

        Returns:
            indices (list): Lista de índices (enteros) de las imágenes más similares en el conjunto db_feat.
            scores (list): Lista de puntajes de similitud correspondientes.
        """
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        results = []
        for idx, feat in enumerate(db_feat):
            # Aplicar el filtro si está definido.
            if self.filter_func is not None:
                filtered_query_feat, filtered_refer_feat, filtered_scores = self.filter_func(query_feat, feat)
            else:
                filtered_query_feat, filtered_refer_feat, filtered_scores = query_feat, feat, None

            score = self.similarity(filtered_query_feat, filtered_refer_feat, filtered_scores)
            results.append((idx, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        top_results = results[:n_neighbors]
        indices = [r[0] for r in top_results]
        scores = [r[1] for r in top_results]
        return indices, scores

    def _get_abs_path(self, rel_path):
        """
        Convierte una ruta relativa en una ruta absoluta respecto al directorio raíz del proyecto.
        Se asume que este archivo está en 'image_retrieval/local_features', y que el root es dos niveles arriba.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.normpath(os.path.join(project_root, rel_path))