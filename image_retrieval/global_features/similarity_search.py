from sklearn.metrics import pairwise_distances
import numpy as np

class kNNSimilarity:
    """
    Realiza búsqueda de similitud global utilizando k-Nearest Neighbors (kNN) con scikit-learn.

    Calcula la matriz de distancias entre múltiples queries y la base de datos, y devuelve los
    índices ordenados por similitud (menor distancia → mayor similitud), siguiendo el formato 
    requerido por el benchmark Revisiting Oxford and Paris.
    """

    def __init__(self, config=None):
        """
        Args:
            config (dict, opcional): Configuración con:
                - "n_neighbors" (int o str): Número de vecinos a devolver ("all" para todos).
                - "metric" (str): Métrica de distancia (por defecto "euclidean").
                - "algorithm" (str): Algoritmo de búsqueda (por defecto "auto").
        """
        if config is None:
            config = {}
        self.n_neighbors = config.get("n_neighbors", 5)
        self.metric = config.get("metric", "euclidean")
        self.algorithm = config.get("algorithm", "auto")

    def search(self, query_feat, db_feat, n_neighbors=None):
        """
        Realiza búsqueda entre queries y la base usando distancias kNN.

        Nota:
            `query_feat` debe tener siempre forma (q, d_global), incluso si q = 1.
            Si se pasa una única imagen como vector (d_global,), se debe hacer reshape explícito
            a (1, d_global) antes de llamar a esta función.

        Args:
            query_feat (np.ndarray): Matriz (q, d_global), donde cada fila es una query.
            db_feat (np.ndarray): Matriz (N, d_global) con descriptores de la base de datos.
            n_neighbors (int o str, opcional): Número de vecinos a devolver. Si es "all",
                                               se devuelven todas las entradas ordenadas.

        Returns:
            ranks (np.ndarray): Matriz (N, q), donde cada columna contiene los índices de la base
                                ordenados por distancia creciente (mayor similitud).
            scores (np.ndarray): Matriz (N, q), con los valores negativos de las distancias ordenadas,
                                 de modo que valores mayores implican más similitud.
        """
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        return_all = isinstance(n_neighbors, str) and n_neighbors.lower() == "all"

        dists = pairwise_distances(query_feat, db_feat, metric=self.metric)  # (q, N)
        dists_T = dists.T  # (N, q)

        ranks = np.argsort(dists_T, axis=0)  # (N, q)
        scores = -dists_T  # Convertir a similitud

        if return_all:
            return ranks, np.take_along_axis(scores, ranks, axis=0)
        else:
            return ranks[:n_neighbors], np.take_along_axis(scores, ranks[:n_neighbors], axis=0)

class DotProductSimilarity:
    """
    Realiza búsqueda de similitud global usando producto punto como métrica de similitud.

    Calcula la matriz de similitudes entre múltiples queries y la base de datos (coseno si están normalizados),
    y devuelve los índices ordenados por similitud descendente, en formato compatible con el benchmark
    Revisiting Oxford and Paris.
    """

    def __init__(self, config=None):
        """
        Args:
            config (dict, opcional): Configuración con:
                - "n_neighbors" (int o str): Número de vecinos a devolver ("all" para todos).
        """
        if config is None:
            config = {}
        self.n_neighbors = config.get("n_neighbors", 5)

    def search(self, query_feat, db_feat, n_neighbors=None):
        """
        Realiza búsqueda entre queries y la base usando producto punto como similitud.

        Nota:
            `query_feat` debe tener siempre forma (q, d_global), incluso si q = 1.
            Si se pasa una única imagen como vector (d_global,), se debe hacer reshape explícito
            a (1, d_global) antes de llamar a esta función.

        Args:
            query_feat (np.ndarray): Matriz (q, d_global), donde cada fila es una query.
            db_feat (np.ndarray): Matriz (N, d_global) con descriptores de la base de datos.
            n_neighbors (int o str, opcional): Número de vecinos a devolver. Si es "all",
                                               se devuelven todas las entradas ordenadas.

        Returns:
            ranks (np.ndarray): Matriz (N, q), donde cada columna contiene los índices de la base
                                ordenados por similitud descendente.
            scores (np.ndarray): Matriz (N, q), con los valores del producto punto correspondientes,
                                 ordenados por similitud.
        """
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        return_all = isinstance(n_neighbors, str) and n_neighbors.lower() == "all"

        Q = query_feat / np.linalg.norm(query_feat, axis=1, keepdims=True)  # (q, d)
        X = db_feat / np.linalg.norm(db_feat, axis=1, keepdims=True)        # (N, d)

        similarities = np.dot(X, Q.T)  # (N, q)
        ranks = np.argsort(-similarities, axis=0)  # (N, q)

        if return_all:
            return ranks, np.take_along_axis(similarities, ranks, axis=0)
        else:
            return ranks[:n_neighbors], np.take_along_axis(similarities, ranks[:n_neighbors], axis=0)