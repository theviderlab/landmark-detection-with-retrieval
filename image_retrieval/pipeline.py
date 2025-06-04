import os
import yaml
import logging
import numpy as np
import pickle
from tqdm import tqdm
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import copy
import json

# Importar funciones de selección
from image_retrieval.select_model import select_model
from image_retrieval.select_extractors import select_local_extractor, select_global_extractor
from image_retrieval.select_aggregator import select_feature_aggregator
from image_retrieval.select_similarity import select_local_similarity, select_global_similarity
from image_retrieval.select_refiner import select_refiner

class ImageRetrievalPipeline:
    def __init__(self, config_file=None):
        """
        Inicializa el pipeline de image retrieval cargando la configuración desde YAML,
        guardando los parámetros necesarios y realizando las inicializaciones pertinentes.

        Se esperan los siguientes bloques en el YAML:
          - General: name, db_directory, embeddings_file, vocabulary_file.
          - preprocess_config: (por ejemplo, resize, use_model_preprocess).
          - feature_extraction_config: que contiene:
              - model_config: parámetros del modelo agrupados.
              - global_extractor_config.
              - local_extractor_config.
              - feature_aggregation_config.
          - similarity_search_config: que contiene:
              - local_similarity_config: configuración para la búsqueda de similitud local, con opciones
                como "filter", "method" y "n_neighbors".
              - global_similarity_config: configuración para la búsqueda de similitud global.
          - refine_results_config: método de refinamiento.
        """
        # Configuración del logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Cargar configuración desde YAML
        if config_file:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
        
        # Parámetros generales
        self.pipeline_name = self.config.get("name", "default_pipeline")
        self.db_directory = self.config.get("db_directory", None)
        if self.db_directory is None:
            raise ValueError("La configuración debe incluir la clave 'db_directory'.")
        self.embeddings_file = self.config.get("embeddings_file", os.path.join(self.db_directory, f"{self.pipeline_name}_embeddings.pkl"))
        
        # Configuración de augmentations
        self.augmentation_config = self.config.get("augmentation_config", {})
        self.use_multiscale = self.augmentation_config.get("use_multiscale", False)
        self.scale_list = self.augmentation_config.get("scale_list", [1.0])

        # Configuración de preprocesado
        preprocess_cfg = self.config.get("preprocess_config", {})
        self.image_size = tuple(preprocess_cfg.get("resize", (224, 224)))
        self.use_model_preprocess = preprocess_cfg.get("use_model_preprocess", True)
        
        # Configuración de extracción de características
        feat_extraction_cfg = self.config.get("feature_extraction_config", {})
        # Configuración del modelo agrupado
        model_config = feat_extraction_cfg.get("model_config", {})
        # Configuraciones para extractores y agregador
        self.global_extractor_config = feat_extraction_cfg.get("global_extractor_config", {})
        self.local_extractor_config = feat_extraction_cfg.get("local_extractor_config", {})
        self.feature_aggregation_config = feat_extraction_cfg.get("feature_aggregation_config", {})
                
        # Configuración de búsqueda de similitud: se dividen en local y global.
        similarity_search_cfg = self.config.get("similarity_search_config", {})
        self.local_similarity_config = similarity_search_cfg.get("local_similarity_config")
        self.global_similarity_config = similarity_search_cfg.get("global_similarity_config")
        
        # Configuración de refinamiento de resultados
        self.refine_results_cfg = self.config.get("refine_results_config", {})
        
        # Seleccionar modelo y función de preprocesado (usando select_model.py)
        self.model, self.preprocess_func = select_model(model_config, self.use_model_preprocess)
        
        # Seleccionar extractores de características (usando select_extractors.py)
        self.local_extractor = select_local_extractor(self.model, self.local_extractor_config)
        self.global_extractor = select_global_extractor(self.model, self.global_extractor_config)
        # Se debe definir extractor local, global o ambos. Si ambos son None, se lanza un error.
        if self.local_extractor is None and self.global_extractor is None:
            raise ValueError("Se debe definir extractor local, global o ambos en la configuración.")
        
        # Seleccionar el agregador de características (usando select_aggregator.py)
        self.feature_aggregation = select_feature_aggregator(self.feature_aggregation_config)
        
        # Seleccionar el refinador de resultados
        self.result_refiner = select_refiner(self.refine_results_cfg)

        # Inicialización de atributos de la base de datos
        self.db_embeddings = None
        
        self.logger.info(f"Pipeline '{self.pipeline_name}' inicializado con modelo {model_config.get('model_type', 'desconocido')}.")

    def load_image(self, image_path):
        """Carga una imagen desde la ruta dada y la convierte a RGB."""
        try:
            img = Image.open(image_path).convert('RGB')
            return img
        except Exception as e:
            print(f"Error cargando la imagen {image_path}: {e}")
            return None

    def augmentations(self, img_list):
        """
        Aplica augmentations sobre una lista de imágenes PIL.
        Devuelve una lista expandida de imágenes PIL + el valor aug por imagen.

        Args:
            img_list (list[PIL.Image]): Lista de imágenes PIL

        Returns:
            tuple:
                - list[PIL.Image]: Lista aumentada (batch * aug)
                - int: Número de augmentations por imagen
        """
        from image_retrieval.preprocess.resize_utils import generate_scaled_images

        all_augmented = []
        aug = 1

        if self.use_multiscale:
            for im in img_list:
                all_augmented.extend(generate_scaled_images(im, self.scale_list))
            aug = len(self.scale_list)
        else:
            all_augmented = img_list

        return all_augmented, aug  # lista de PIL.Image

    def preprocess_image(self, img_list):
        processed = []
        for im in img_list:
            resized = im.resize(self.image_size)
            arr = keras_image.img_to_array(resized)
            arr = self.preprocess_func(np.expand_dims(arr, axis=0))  # (1, H, W, C)
            processed.append(arr[0])
        return np.stack(processed, axis=0)  # (n, H, W, C)

    def extract_features(self, preprocessed_img, aug=1):
        """
        Extrae las características de una o varias imágenes preprocesadas utilizando los extractores configurados.

        Este método admite augmentations por imagen. Si la entrada contiene augmentations, estas deben estar
        agrupadas secuencialmente. El parámetro 'aug' especifica cuántas augmentations hay por imagen.

        Args:
            preprocessed_img (np.ndarray): Tensor de forma (n, H, W, C), donde n = batch_size * aug.
            aug (int): Número de augmentations por imagen. Por defecto es 1.

        Returns:
            tuple:
                - local_features (list[np.ndarray]): Lista de arrays con forma (batch_size, d_local),
                donde cada entrada corresponde a una imagen y contiene sus características locales.
                Actualmente se extrae solo de la primera augmentación.
                
                - global_features (np.ndarray): Array con forma (batch_size, d_global),
                donde d_global es la dimensión del descriptor global fusionado a partir de las augmentations.
        """
        local_features = None
        global_features = None
        batch_size = preprocessed_img.shape[0] // aug

        if self.local_extractor is not None:
            local_features = []
            for i in range(batch_size):
                img = preprocessed_img[i * aug : (i * aug) + 1]  # solo la primera augmentación
                feats = self.local_extractor.extract(img)
                local_features.append(feats)

        if self.global_extractor is not None:
            global_features = self.global_extractor.extract(preprocessed_img, aug=aug)

        return local_features, global_features

    def aggregate_features(self, local_features, global_features):
        """
        Agrega las características locales y globales para obtener un vector 
        de características agregadas (aggregated_features).

        Args:
            local_features (list[np.ndarray] or None): Lista de arrays de características locales
                                                    (M_i, d_local) por imagen, o None.
            global_features (np.ndarray or None): Array (batch_size, d_global), o None.

        Returns:
            np.ndarray: Array (batch_size, d_aggregated). 
                        - Si hay agregador: resultado del método .aggregate().
                        - Si no hay agregador: se devuelve global_features directamente.
        
        Raises:
            ValueError: Si ambos local_features y global_features son None y no hay agregador definido.
        """
        if self.feature_aggregation is not None:
            return self.feature_aggregation.aggregate(local_features, global_features)
        elif global_features is not None:
            return global_features
        else:
            raise ValueError("No se puede realizar la agregación: global_features es None y no hay agregador definido.")

    def search_similar(self, local_features, global_features):
        """
        Busca las imágenes más similares en la base utilizando los métodos de similitud
        configurados para características locales y globales.

        Soporta tanto una única imagen de consulta como múltiples imágenes en batch.

        Args:
            local_features: Características locales de las imágenes de consulta.
                - Si es una única imagen: np.ndarray de forma (m, d_local)
                - Si son múltiples imágenes: lista de arrays, uno por imagen (longitud q)

            global_features: Características globales (aggregated_features) de las consultas.
                - Una imagen: np.ndarray de forma (1, d)
                - Varias imágenes: np.ndarray de forma (q, d)

        Returns:
            dict: Resultados de búsqueda con las siguientes claves:
                - "local":
                    - Si es una imagen: tuple (indices: np.ndarray, scores: np.ndarray)
                    - Si son múltiples imágenes: tuple (indices_list: list[np.ndarray], scores_list: list[np.ndarray])
                - "global":
                    - Siempre: tuple (ranks: np.ndarray, scores: np.ndarray)
                        donde:
                            - ranks tiene forma (N, q), con los índices ordenados por similitud (mayor a menor)
                            - scores tiene forma (N, q), con los valores de similitud correspondientes
                            - q = número de imágenes de consulta
                            - N = número de imágenes en la base
        """

        self.local_similarity = select_local_similarity(self.local_similarity_config)
        self.global_similarity = select_global_similarity(self.global_similarity_config)

        if local_features is None and global_features is None:
            raise ValueError("Se debe proporcionar al menos uno de los vectores: local_features o global_features.")

        db_local_features = self.db_embeddings.get("local_features", None)
        db_global_features = self.db_embeddings.get("global_features", None)

        local_active = self.local_similarity is not None
        global_active = self.global_similarity is not None

        if not local_active and not global_active:
            raise ValueError("No se ha definido un método de búsqueda de similitud para características locales ni globales.")

        results = {"local": None, "global": None}

        # --- Local ---
        if local_active:
            # Si es una lista de múltiples imágenes (e.g., durante evaluación por batch)
            if isinstance(local_features, list) and isinstance(local_features[0], np.ndarray):
                indices_list = []
                scores_list = []
                for lf in local_features:
                    idx, sc = self.local_similarity.search(lf, db_local_features)
                    indices_list.append(idx)
                    scores_list.append(sc)
                results["local"] = (indices_list, scores_list)
            else:
                # Solo una imagen
                indices_local, scores_local = self.local_similarity.search(local_features, db_local_features)
                results["local"] = (indices_local, scores_local)

        # --- Global ---
        if global_active:
            indices_global, scores_global = self.global_similarity.search(global_features, db_global_features)
            results["global"] = (indices_global, scores_global)  # (N, q)

        return results

    def refine_results(self, results):
        """
        Aplica el método de refinamiento configurado sobre los resultados globales.

        Si no hay refinador configurado, simplemente devuelve el resultado global tal como está.

        Args:
            results (dict): Diccionario que contiene al menos:
                - "global": (ranks, scores)
                - "query_global_raw": descriptores globales de la(s) imagen(es) de consulta

        Returns:
            tuple: (ranks, scores), refinados si aplica, originales si no
        """
        if self.result_refiner is None:
            return results["global"]

        ranks, scores = results["global"]
        Q = results.get("query_global_raw")
        X = self.db_embeddings.get("global_features")

        if Q is None or X is None:
            raise ValueError("Faltan descriptores globales para aplicar reranking.")

        new_ranks = self.result_refiner.refine(X_np=X, Q_np=Q, ranks_np=ranks)
        return new_ranks, scores

    def get_features(self, query_image_path):
        """
        Extrae características locales y globales (agregadas) de una o más imágenes de consulta.

        Este método admite tanto una única imagen como una lista de imágenes. En ambos casos:
        - Aplica augmentations (por ejemplo, escalado multiscale si está configurado).
        - Preprocesa las imágenes según el modelo backbone.
        - Extrae características locales (si está configurado un extractor local).
        - Extrae características globales y las agrega (si hay un extractor global y/o agregador).

        Args:
            query_image_path (str or list[str]): Ruta o lista de rutas a imágenes.

        Returns:
            Si se pasa una sola imagen:
                tuple: (local_features, aggregated_features)
                    - local_features (np.ndarray or None): Array (M, d_local) o None si no hay extractor local.
                    - aggregated_features (np.ndarray or None): Array (1, d_global) o None si no hay extractor global.

            Si se pasa una lista de imágenes:
                tuple: (list_local_features, aggregated_features)
                    - list_local_features (list[np.ndarray] or None): Lista con un array por imagen, o None si no hay extractor local.
                    - aggregated_features (np.ndarray or None): Array (batch_size, d_global) o None si no hay extractor global.
        """
        # Detectar si es entrada única o lista
        if isinstance(query_image_path, str):
            query_image_path = [query_image_path]

        # 1. Cargar imágenes PIL
        images = []
        for path in query_image_path:
            img = self.load_image(path)
            if img is not None:
                images.append(img)
            else:
                print(f"Error al cargar la imagen: {path}")

        if not images:
            return None, None

        # 2. Aplicar augmentations (multiscale, etc.)
        augmented, aug = self.augmentations(images)

        # 3. Preprocesamiento
        preprocessed = self.preprocess_image(augmented)

        # 4. Extracción de características
        local_feats, global_feats = self.extract_features(preprocessed, aug=aug)

        # 5. Agregación
        aggregated_feats = self.aggregate_features(local_feats, global_feats)

        # 6. Formato de retorno
        return local_feats, aggregated_feats

    def run(self, query_image_path):
        """
        Ejecuta el pipeline completo para una o más imágenes de consulta.

        Pasos realizados:
        1. Carga y preprocesamiento de la(s) imagen(es).
        2. Extracción de características locales y globales.
        3. Búsqueda de similitud utilizando los métodos configurados.
        4. Refinamiento de resultados si está configurado.

        Args:
            query_image_path (str o list[str]): Ruta o lista de rutas de imágenes de consulta.

        Returns:
            tuple:
                - ranks (np.ndarray): Índices ordenados por similitud (N, q)
                - scores (np.ndarray): Valores de similitud (N, q)
        """
        local_features, aggregated_features = self.get_features(query_image_path)

        self.load_db_embeddings()
        self.exclude_imlist_from_db_embeddings(query_image_path)

        results = self.search_similar(local_features, aggregated_features)

        # Guardar descriptores originales de la query para posibles refinadores
        results["query_local_raw"] = local_features
        results["query_global_raw"] = aggregated_features

        results = self.refine_results(results)

        return results

    # Métodos auxiliares

    def evaluate(self, datasets, save_to_json=True):
        from benchmark.revisitop.download import download_datasets
        from benchmark.revisitop.dataset import configdataset
        from benchmark.revisitop.evaluate import compute_map
        from image_retrieval.metrics.voting import compute_voting_precision
        import time
        from pympler import asizeof
        import pandas as pd
        from datetime import datetime

        # Validar si el pipeline ya fue evaluado
        eval_path = os.path.join("evaluations", "evaluations.json")
        if save_to_json and os.path.exists(eval_path):
            with open(eval_path, "r") as f:
                previous_evals = json.load(f)
            existing_names = [e['pipeline'].get('name') for e in previous_evals if 'pipeline' in e]
            if self.config.get("name") in existing_names:
                print(f"⚠️  El pipeline '{self.config.get('name')}' ya fue evaluado. Cancelando ejecución.")
                return None, None
            
        # Ruta donde se almacenan los datasets
        data_root = self._get_abs_path("../assets/database")
        download_datasets(data_root)

        resumen = []   # Tabla resumen de resultados
        evaluations = []  # Lista con las evaluaciones por dataset
        ks = [1, 5, 10, 20]  # Valores de k a evaluar

        for dataset in datasets:
            print(f"\n================ Evaluando {dataset.upper()} ================\n")
            cfg = configdataset(dataset, os.path.join(data_root, 'datasets'))

            dataset_dir = cfg['dir_images']
            embeddings_dir = os.path.join(cfg['dir_data'], 'embeddings')
            os.makedirs(embeddings_dir, exist_ok=True)
            embeddings_file = os.path.join(embeddings_dir, f"{self.pipeline_name}.pkl")

            self.build_database(db_directory=dataset_dir, embeddings_file=embeddings_file)

            mem_global = self.db_embeddings["global_features"].nbytes
            mem_local = asizeof.asizeof(self.db_embeddings["local_features"])
            mem_paths = asizeof.asizeof(self.db_embeddings["paths"])
            n_images = len(self.db_embeddings["paths"])
            mem_total_bytes = mem_global + mem_local + mem_paths
            mem_total_mb = mem_total_bytes / 1024**2
            mem_per_image_bytes = mem_total_bytes / n_images

            q_paths = [cfg['qim_fname'](cfg, i) for i in range(cfg['nq'])]

            start_time = time.perf_counter()
            ranks, scores = self.run(q_paths)
            inference_time = time.perf_counter() - start_time
            avg_inference_time = inference_time / len(q_paths)

            ranks = self.remap_ranks_to_imlist_order(ranks, cfg['imlist'])

            gnd = cfg['gnd']

            gnd_easy = [{'ok': np.concatenate([g['easy']]),
                        'junk': np.concatenate([g['junk'], g['hard']])} for g in gnd]
            gnd_medium = [{'ok': np.concatenate([g['easy'], g['hard']]),
                        'junk': np.concatenate([g['junk']])} for g in gnd]
            gnd_hard = [{'ok': np.concatenate([g['hard']]),
                        'junk': np.concatenate([g['junk'], g['easy']])} for g in gnd]

            mapE, apsE, mprE, prsE = compute_map(ranks, gnd_easy, ks)
            mapM, apsM, mprM, prsM = compute_map(ranks, gnd_medium, ks)
            mapH, apsH, mprH, prsH = compute_map(ranks, gnd_hard, ks)

            image_to_place = cfg['image_to_place']
            vp = {k: compute_voting_precision(ranks, scores, q_paths, cfg['imlist'], image_to_place, top_k=k) for k in ks}

            resumen.append({
                "Dataset": dataset,
                "mAP (Easy)": round(mapE * 100, 2),
                "mAP (Med)": round(mapM * 100, 2),
                "mAP (Hard)": round(mapH * 100, 2),
                **{f"mP@{k}": round(mprE[i] * 100, 2) for i, k in enumerate(ks)},
                **{f"VP@{k}": round(vp[k] * 100, 2) for k in ks},
                "Tiempo total (s)": round(inference_time, 2),
                "Tiempo/query (s)": round(avg_inference_time, 4),
                "Mem total (MB)": round(mem_total_mb, 2),
                "Mem por imagen (bytes)": round(mem_per_image_bytes, 2)
            })

            from datetime import datetime

            test_datetime = datetime.now().isoformat()

            evaluations.append({
                'dataset': dataset,
                'map': {'easy': mapE, 'medium': mapM, 'hard': mapH},
                'mpr': {'easy': mprE.tolist(), 'medium': mprM.tolist(), 'hard': mprH.tolist()},
                'vp': vp,
                'per_query_seconds': avg_inference_time,
                'total_inference_seconds': inference_time,
                'memory': {
                    'global_bytes': mem_global,
                    'local_bytes': mem_local,
                    'paths_bytes': mem_paths,
                    'total_mb': mem_total_mb,
                    'per_image_bytes': mem_per_image_bytes
                },
                'ks': ks,
                'num_queries': len(q_paths),
                'num_database_images': n_images
            })

        df = pd.DataFrame(resumen)
        print("\n============== RESUMEN FINAL ==============")
        print(df.to_string(index=False))

        if save_to_json:
            os.makedirs("evaluations", exist_ok=True)
            eval_path = os.path.join("evaluations", "evaluations.json")

        if os.path.exists(eval_path):
            with open(eval_path, "r") as f:
                existing = json.load(f)
        else:
            existing = []

        existing.append({
            'datetime': test_datetime,
            'pipeline': self.config,
            'evaluations': evaluations
        })

        with open(eval_path, "w") as f:
            json.dump(existing, f, indent=2)

        return df, evaluations

    def compute_map_at_k(self, evaluation_results, df_full, k): 
        average_precisions = []
        for query_eval in evaluation_results:
            query_place = query_eval["query_place"]
            retrieved_places = query_eval["retrieved_places"][:k]  # Utilizamos directamente la lista aplanada
            num_relevant = len(df_full[(df_full["place"] == query_place) & (df_full["type"].str.lower() == "good")])
            if num_relevant == 0:
                average_precisions.append(0)
                continue

            num_hits = 0
            precision_sum = 0.0
            for i, retrieved_place in enumerate(retrieved_places, start=1):
                if retrieved_place == query_place:
                    num_hits += 1
                    precision_sum += num_hits / i
            AP = precision_sum / num_relevant
            average_precisions.append(AP)

        mAP = sum(average_precisions) / len(average_precisions) if average_precisions else 0
        return mAP

    def display_evaluation_results(self, evaluation_results, num_results=5): 
        import matplotlib.pyplot as plt

        n_queries = len(evaluation_results)
        ncols = 1 + num_results  # Una columna para la query y num_results para los resultados
        fig, axes = plt.subplots(n_queries, ncols, figsize=(4*ncols, 4*n_queries))

        if n_queries == 1:
            axes = [axes]

        for i, eval_res in enumerate(evaluation_results):
            query_path = eval_res["query_image"]
            query_place = eval_res["query_place"]
            query_img = self.load_image(query_path)
            ax = axes[i][0] if n_queries > 1 else axes[0]
            ax.imshow(query_img)
            ax.set_title(f"Query\nPlace: {query_place}")
            ax.axis("off")
            
            db_img_paths = eval_res["db_img_paths"]
            scores = eval_res["scores"]
            retrieved_places = eval_res["retrieved_places"]
            for j in range(num_results):
                ax_index = j + 1  # Las columnas de resultados comienzan en 1
                if j < len(db_img_paths):
                    db_path = db_img_paths[j]
                    score = scores[j]
                    retrieved_place = retrieved_places[j]
                    result_img = self.load_image(db_path)
                    ax = axes[i][ax_index] if n_queries > 1 else axes[ax_index]
                    ax.imshow(result_img)
                    ax.set_title(f"Score: {score:.2f}\nPlace: {retrieved_place}")
                    ax.axis("off")
                else:
                    ax = axes[i][ax_index] if n_queries > 1 else axes[ax_index]
                    ax.axis("off")

        plt.tight_layout()
        plt.show()

    def build_database(self, force_rebuild=False, batch_size=32, db_directory=None, embeddings_file=None):
        """
        Construye la base de datos de embeddings en batches, aprovechando el soporte para procesamiento vectorizado.

        Args:
            force_rebuild (bool): Si True, se ignora el archivo existente y se reconstruye desde cero.
            batch_size (int): Cantidad de imágenes a procesar simultáneamente.
            db_directory (str, optional): Ruta al directorio con las imágenes de la base. Si None, se usa self.db_directory.
            embeddings_file (str, optional): Ruta al archivo donde guardar los embeddings. Si None, se usa self.embeddings_file.
        """
        db_directory = db_directory or self.db_directory
        embeddings_file = embeddings_file or self.embeddings_file

        abs_db_directory = self._get_abs_path(db_directory)
        abs_embeddings_file = self._get_abs_path(embeddings_file)

        if os.path.exists(abs_embeddings_file) and not force_rebuild:
            with open(abs_embeddings_file, 'rb') as f:
                final_db = pickle.load(f)
            print(f"Se cargó la base de datos de embeddings existente desde {abs_embeddings_file}.")
            local_list = final_db.get("local_features", [])
            global_array = final_db.get("global_features", None)
            paths = final_db.get("paths", [])
        else:
            local_list = []
            global_array = None
            paths = []
            print(f"No se encontró el archivo de embeddings o se fuerza su reconstrucción. Se creará uno nuevo: {abs_embeddings_file}.")

        image_files = [
            os.path.join(abs_db_directory, file)
            for file in os.listdir(abs_db_directory)
            if file.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        image_files = [img for img in image_files if img not in paths]
        total_images = len(image_files)
        print(f"Se procesarán {total_images} imágenes nuevas en {abs_db_directory}.")

        new_embeddings_count = 0

        for i in tqdm(range(0, total_images, batch_size), desc="Procesando imágenes", unit="batch"):
            batch_paths = image_files[i:i + batch_size]
            local_feats, global_feats = self.get_features(batch_paths)

            if local_feats is None and global_feats is None:
                continue

            for j, path in enumerate(batch_paths):
                local_feat = local_feats[j] if local_feats else None
                if local_feat is not None:
                    local_list.append(local_feat)
                paths.append(path)
                new_embeddings_count += 1

            if global_feats is not None:
                if global_array is None:
                    global_array = global_feats
                else:
                    global_array = np.concatenate([global_array, global_feats], axis=0)

            if new_embeddings_count % 1000 == 0:
                final_db = {
                    "local_features": local_list,
                    "global_features": global_array,
                    "paths": paths
                }
                with open(abs_embeddings_file, 'wb') as f:
                    pickle.dump(final_db, f)

        final_db = {
            "local_features": local_list,
            "global_features": global_array,
            "paths": paths
        }
        with open(abs_embeddings_file, 'wb') as f:
            pickle.dump(final_db, f)

        print(f"Base de datos construida con {len(paths)} imágenes.")
        self.db_embeddings = final_db

    def build_vocabulary(self, num_clusters=1000):
        """
        Construye un vocabulario de características (visual words) a partir de las local_features 
        almacenadas en self.db_embeddings. Se espera que self.db_embeddings tenga la siguiente estructura:
        
            {
                "local_features": list of numpy arrays, each array of shape (m_i, d_local),
                "global_features": numpy array of shape (N, d_global),
                "paths": list of str, length N
            }
        
        Para cada imagen:
        - Se asume que cada elemento en "local_features" es un array de forma (m_i, d_local).
        - Se concatenan todos los vectores (de todas las imágenes) en una única matriz de forma (total_vectors, d_local)
            para entrenar MiniBatchKMeans.
        
        Se calcula el vector idf para cada cluster usando:
            idf[c] = log(K / (nc + epsilon))
        donde K es el número total de imágenes y nc es el número de imágenes en las que aparece el cluster.
        
        El vocabulario (diccionario con "kmeans" e "idf") se guarda en el path definido en:
            self.local_similarity_config.get('vocabulary_file')
        
        Returns:
            vocab (dict): Diccionario con el vocabulario construido.
        """
        # Obtener el path para guardar el vocabulario y convertirlo a absoluto
        vocabulary_file_rel = self.local_similarity_config.get('vocabulary_file', None)
        if vocabulary_file_rel is None:
            raise ValueError("Se debe especificar 'vocabulary_file' en la configuración de local_similarity_config.")
        vocabulary_file = self._get_abs_path(vocabulary_file_rel)

        # Cargar la base de datos de embeddings.
        self.load_db_embeddings()
        
        # Extraer las local_features y los paths.
        # Se asume que "local_features" es una lista de arrays, cada uno de forma (m_i, d_local).
        local_feats = self.db_embeddings["local_features"]
        paths = self.db_embeddings["paths"]
        N = len(local_feats)
        print(f"Utilizando {N} imágenes para construir el vocabulario.")
        
        # Inicializar lista para acumular todos los vectores y un diccionario para registrar las asignaciones.
        pooled_vectors_list = []
        image_assignments = {}
        current_index = 0
        for i in range(N):
            # Cada imagen ya tiene sus local features en forma (m_i, d_local), por lo que no se necesita transponer.
            img_local = local_feats[i]
            M = img_local.shape[0]  # Número de vectores pooled para la imagen
            pooled_vectors_list.append(img_local)
            image_assignments[paths[i]] = list(range(current_index, current_index + M))
            current_index += M

        # Concatenar todas las matrices en una única matriz de forma (total_vectors, d_local)
        pooled_vectors = np.vstack(pooled_vectors_list)
        print("Pooled vectors shape:", pooled_vectors.shape)
        
        # Entrenar clustering usando MiniBatchKMeans.
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(pooled_vectors)
        
        # Calcular la presencia de cada cluster en cada imagen.
        cluster_presence = np.zeros(num_clusters)
        for indices in image_assignments.values():
            if len(indices) == 0:
                continue
            img_vectors = pooled_vectors[indices]
            clusters = kmeans.predict(img_vectors)
            unique_clusters = np.unique(clusters)
            cluster_presence[unique_clusters] += 1
        
        epsilon = 1e-6
        idf = np.log(N / (cluster_presence + epsilon))
        
        vocab = {"kmeans": kmeans, "idf": idf}
        
        # Guardar el vocabulario en el archivo especificado.
        with open(vocabulary_file, 'wb') as f:
            pickle.dump(vocab, f)
        print(f"Vocabulario construido y guardado en {vocabulary_file}.")

        return vocab
        
    def load_db_embeddings(self):
        """
        Carga el archivo de embeddings en self.db_embeddings (si aún no está cargado) y verifica que la estructura
        sea la esperada:
        
            {
                "local_features": list (puede estar vacía),
                "global_features": np.ndarray o None,
                "paths": list of str, length N
            }

        Al menos uno de los dos (local o global) debe tener información. Y si alguno está presente,
        su longitud debe coincidir con la de paths.

        Raises:
            ValueError: Si el archivo no existe o la estructura no es válida.
        """
        abs_embeddings_file = self._get_abs_path(self.embeddings_file)

        if self.db_embeddings is None:
            if not self.embeddings_file or not os.path.exists(abs_embeddings_file):
                raise ValueError(f"El embeddings_file '{abs_embeddings_file}' no está configurado o no existe.")
            with open(abs_embeddings_file, 'rb') as f:
                self.db_embeddings = pickle.load(f)

        db = self.db_embeddings

        if not isinstance(db, dict) or not all(k in db for k in ["local_features", "global_features", "paths"]):
            raise ValueError("El diccionario de embeddings debe contener las claves 'local_features', 'global_features' y 'paths'.")

        N = len(db["paths"])
        local_feats = db["local_features"]
        global_feats = db["global_features"]

        local_ok = isinstance(local_feats, list) and len(local_feats) == N
        global_ok = isinstance(global_feats, np.ndarray) and global_feats.shape[0] == N

        if not local_ok and not global_ok:
            raise ValueError(
                f"Inconsistencia en el número de elementos:\n"
                f"- local_features: {len(local_feats) if isinstance(local_feats, list) else 'inválido'}\n"
                f"- global_features: {global_feats.shape[0] if isinstance(global_feats, np.ndarray) else 'inválido o None'}\n"
                f"- paths: {N}\n"
                f"Debe haber al menos uno de los dos (local o global) con {N} elementos."
            )
            
    def exclude_imlist_from_db_embeddings(self, imlist):
        """
        Elimina de self.db_embeddings todas las imágenes cuyo nombre de archivo (sin ruta)
        aparece en `imlist`.

        Args:
            imlist (list of str): Lista de rutas o nombres de archivo (por ejemplo: ['.../all_souls_000002.jpg'])
        """
        exclude_names = set(os.path.basename(name) for name in imlist)
        db = self.db_embeddings

        filtered = [(i, path) for i, path in enumerate(db["paths"]) if os.path.basename(path) not in exclude_names]
        indices = [i for i, _ in filtered]
        filtered_paths = [db["paths"][i] for i in indices]

        new_db = {"paths": filtered_paths}

        if isinstance(db["local_features"], list) and len(db["local_features"]) == len(db["paths"]):
            new_db["local_features"] = [db["local_features"][i] for i in indices]
        else:
            new_db["local_features"] = db["local_features"]

        if isinstance(db["global_features"], np.ndarray) and db["global_features"].shape[0] == len(db["paths"]):
            new_db["global_features"] = db["global_features"][indices]
        else:
            new_db["global_features"] = db["global_features"]

        self.db_embeddings = new_db

    def remap_ranks_to_imlist_order(self, ranks, imlist):
        """
        Reordena las filas de `ranks` para que correspondan al orden de `imlist`.

        Esto es necesario porque `self.db_embeddings["paths"]` puede estar en un orden
        diferente al que espera el benchmark (cfg['imlist']).

        Args:
            ranks (np.ndarray): Matriz (N, nq) con índices de imágenes en el orden de db_embeddings["paths"].
            imlist (list of str): Lista de nombres de archivo sin extensión (e.g. 'all_souls_000000').

        Returns:
            np.ndarray: ranks remapeado, donde cada valor es el índice relativo a `imlist`.
        """
        import os

        # Añadir extensión .jpg a todos los nombres
        imlist_with_ext = [name + ".jpg" for name in imlist]

        # Crear mapa: nombre con extensión → índice en imlist
        name_to_imlist_index = {name: i for i, name in enumerate(imlist_with_ext)}

        db_paths = self.db_embeddings["paths"]

        # Mapeo de índices de la base a su posición en imlist
        db_index_to_imlist_index = [
            name_to_imlist_index[os.path.basename(p)] for p in db_paths
        ]

        # Reasignar los índices en ranks
        ranks_remapped = np.vectorize(lambda idx: db_index_to_imlist_index[idx])(ranks)

        return ranks_remapped

    def display_results(self, query_img_paths, ranks, scores, cfg, gnd, top_k=5):
        """
        Muestra en una grilla las imágenes de consulta y sus top-k resultados,
        marcando los aciertos con borde verde y los errores con rojo.

        Args:
            query_img_paths (list of str): Rutas de imágenes de consulta (en el mismo orden que cfg['qimlist']).
            ranks (np.ndarray): Matriz (N, nq) con los índices (remapeados) de recuperación.
            scores (np.ndarray): Matriz (N, nq) con los scores correspondientes.
            cfg (dict): Configuración del dataset.
            gnd (list of dict): Ground truth por query (cfg['gnd']).
            top_k (int): Número de resultados a mostrar por query.
        """
        import matplotlib.pyplot as plt
        import os

        nq = len(query_img_paths)
        top_k = min(top_k, ranks.shape[0])
        plt.figure(figsize=(3 * (top_k + 1), 4 * nq))

        for i in range(nq):
            # Mostrar imagen de consulta
            plt.subplot(nq, top_k + 1, i * (top_k + 1) + 1)
            plt.imshow(self.load_image(query_img_paths[i]))
            plt.title("Query")
            plt.axis("off")

            # Índices positivos para esta query
            positives = set(gnd[i]['ok'])

            # Mostrar top-k resultados
            for j in range(top_k):
                db_idx = ranks[j, i]
                score = scores[j, i]
                img_path = cfg['im_fname'](cfg, db_idx)
                img_name = os.path.basename(img_path)

                ax = plt.subplot(nq, top_k + 1, i * (top_k + 1) + j + 2)
                ax.imshow(self.load_image(img_path))
                ax.set_title(f"[{db_idx}] {img_name}\nScore: {score:.2f}", fontsize=8)
                ax.axis("off")

                # Marcar borde verde si es acierto, rojo si no
                color = 'lime' if db_idx in positives else 'red'
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(3)

        plt.tight_layout()
        plt.show()

    def _filter_db_embeddings_by_type(self, df, keep_types): 
        """ Filtra self.db_embeddings para conservar únicamente las imágenes cuyo tipo, 
        obtenido del dataframe 'df', esté en keep_types. Si 'other' se encuentra en keep_types, 
        se conservarán las imágenes que no estén presentes en el dataframe.
        Se guarda el self.db_embeddings original en self.original_db_embeddings para poder restaurarlo.

        Args:
            df (pandas.DataFrame): DataFrame con columnas: image_name, label, type, place.
            keep_types (list): Lista de tipos a conservar (e.g., ['query', 'good', 'other']).
        """

        # Guardar el estado original para poder restaurarlo después
        self.original_db_embeddings = copy.deepcopy(self.db_embeddings)

        # Crear un mapeo de image_name a type a partir del dataframe
        df_mapping = df.set_index("image_name")["type"].to_dict()

        filtered_paths = []
        filtered_local_features = []
        filtered_global_features = []

        # Convertir keep_types a minúsculas para comparación
        keep_types_lower = [t.lower() for t in keep_types]

        for i, path in enumerate(self.db_embeddings["paths"]):
            img_name = os.path.basename(path)
            if img_name in df_mapping:
                img_type = df_mapping[img_name].lower()
                if img_type in keep_types_lower:
                    filtered_paths.append(path)
                    filtered_local_features.append(self.db_embeddings["local_features"][i])
                    filtered_global_features.append(self.db_embeddings["global_features"][i])
            else:
                # Imagen no presente en el dataframe
                if "other" in keep_types_lower:
                    filtered_paths.append(path)
                    filtered_local_features.append(self.db_embeddings["local_features"][i])
                    filtered_global_features.append(self.db_embeddings["global_features"][i])

        self.db_embeddings["paths"] = filtered_paths
        self.db_embeddings["local_features"] = filtered_local_features
        self.db_embeddings["global_features"] = np.array(filtered_global_features)
        print(f"db_embeddings filtrado: se han conservado {len(filtered_paths)} imágenes según los tipos {keep_types}.")

    def _restore_db_embeddings(self): 
        """ Restaura self.db_embeddings al estado original guardado previamente con 
        filter_db_embeddings_by_type. 
        """ 
        
        if hasattr(self, "original_db_embeddings"): 
            self.db_embeddings = self.original_db_embeddings 
            print("db_embeddings restaurado al estado original.") 
        else: 
            print("No se encontró un estado original de db_embeddings para restaurar.")

    def _get_abs_path(self, rel_path):
        """
        Convierte una ruta relativa (definida en el YAML) en una ruta absoluta relativa al directorio raíz del proyecto.
        Se asume que el directorio raíz es la carpeta padre del directorio 'image_retrieval'.
        """
        # Suponiendo que __file__ está en 'TFM/image_retrieval', el directorio raíz es 'TFM'
        project_root = os.path.dirname(os.path.abspath(__file__))
        return os.path.normpath(os.path.join(project_root, rel_path))

    def model_summary(self):
        """Muestra un resumen de la arquitectura del modelo utilizado."""
        self.model.summary()

    def grad_cam_embedding(self, image_path, layer=None, intensity=0.6):
        """
        Calcula un mapa de calor similar a Grad-CAM enfocado en el embedding para una imagen.

        Args:
            image_path (str): Ruta de la imagen.
            layer (str): Nombre de la capa convolucional a usar (usa la predefinida si es None).
            intensity (float): Intensidad de la superposición.

        Returns:
            heatmap_colored: Mapa de calor coloreado.
            image_grad: Imagen original con el heatmap superpuesto.
        """
        if layer is None:
            layer = self.default_grad_cam_layer
        img = self.load_image(image_path)
        if img is None:
            print("Error al cargar la imagen para grad_cam_embedding.")
            return None, None
        preprocessed = self.preprocess_image([img])
        last_conv_layer = self.model.get_layer(layer)
        grad_model = tf.keras.models.Model(self.model.inputs, [self.model.output, last_conv_layer.output])
        with tf.GradientTape() as tape:
            embedding, conv_output = grad_model(preprocessed)
            objective = tf.reduce_mean(embedding)
        grads = tape.gradient(objective, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_output = conv_output[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        if tf.reduce_max(heatmap) != 0:
            heatmap /= tf.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        heatmap_resized = cv2.resize(heatmap, (img.size[0], img.size[1]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        image_grad = cv2.addWeighted(img_cv, 1.0, heatmap_colored, intensity, 0)
        return heatmap_colored, image_grad

    def display_grad_cam_embedding(self, image_path, layers=None, intensity=0.4):
        """
        Calcula y muestra en una grilla el Grad-CAM para una o varias capas.
        Cada fila muestra:
          - La imagen original.
          - El mapa de calor.
          - La imagen con Grad-CAM superpuesto.

        Args:
            image_path (str): Ruta de la imagen.
            layers (list or str): Lista o único nombre de capa. Si es None, se usa la predefinida.
            intensity (float): Intensidad de la superposición.
        """
        if layers is None:
            layers = [self.default_grad_cam_layer]
        elif not isinstance(layers, list):
            layers = [layers]
        img = self.load_image(image_path)
        if img is None:
            print("Error al cargar la imagen.")
            return
        n_layers = len(layers)
        fig, axes = plt.subplots(n_layers, 3, figsize=(15, 5 * n_layers))
        if n_layers == 1:
            axes = [axes]
        for i, layer in enumerate(layers):
            heatmap, image_grad = self.grad_cam_embedding(image_path, layer, intensity)
            axes[i][0].imshow(img)
            axes[i][0].set_title(f"Original - {layer}")
            axes[i][0].axis("off")
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            axes[i][1].imshow(heatmap_rgb)
            axes[i][1].set_title(f"Mapa de Calor - {layer}")
            axes[i][1].axis("off")
            image_grad_rgb = cv2.cvtColor(image_grad, cv2.COLOR_BGR2RGB)
            axes[i][2].imshow(image_grad_rgb)
            axes[i][2].set_title(f"Grad-CAM Embedding - {layer}")
            axes[i][2].axis("off")
        plt.tight_layout()
        plt.show()
