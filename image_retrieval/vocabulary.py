# vocabulary.py
import os
import pickle
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

def build_vocabulary(pipeline, vocabulary_file, num_clusters=1000):
    """
    Construye un vocabulario de características (visual words) a partir de las pooled features
    extraídas con el extractor region-based del pipeline.

    Para cada imagen en el directorio pipeline.db_directory se extraen las pooled features (matriz de forma (D, M))
    usando el extractor region-based y se recopilan todos los vectores (total: K x M). Luego se agrupan estos
    vectores en 'num_clusters' clusters usando MiniBatchKMeans.
    Finalmente, se calcula el peso idf para cada visual word según:
         Wc = log(K / nc)
    donde K es el número total de imágenes y nc es el número de imágenes que contienen la visual word c.

    Además, se guarda de forma incremental un archivo temporal para retomar el proceso en caso de fallo.

    Args:
        pipeline: Instancia del pipeline de image retrieval. Se espera que:
                  - pipeline.db_directory sea el path relativo al directorio de imágenes.
                  - pipeline.region_extractor esté instanciado y tenga el método extract(preprocessed_img)
                    que devuelve una matriz de pooled features de forma (D, M).
        vocabulary_file (str): Ruta del archivo donde se guardará el vocabulario (pickle).
        num_clusters (int): Número de clusters (visual words) a generar.

    Returns:
        vocab (dict): Diccionario con el vocabulario, con las claves:
                      - "kmeans": modelo k-means entrenado.
                      - "idf": vector de pesos idf de tamaño (num_clusters,).
    """
    # Construir la ruta absoluta del directorio de imágenes
    base_dir = os.path.dirname(os.path.abspath(__file__))
    abs_db_directory = os.path.normpath(os.path.join(base_dir, pipeline.db_directory))
    
    # Listar todas las imágenes en el directorio
    image_files = [os.path.join(abs_db_directory, file) 
                   for file in os.listdir(abs_db_directory)
                   if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_files)
    print(f"Encontradas {total_images} imágenes en {abs_db_directory}.")

    # Definir el archivo temporal para guardar el avance
    temp_features_file = vocabulary_file + ".tmp"
    
    # Intentar cargar el avance temporal
    if os.path.exists(temp_features_file):
        try:
            with open(temp_features_file, 'rb') as f:
                tmp_data = pickle.load(f)
            pooled_vectors = tmp_data.get("pooled_vectors", [])
            image_assignments = tmp_data.get("image_assignments", {})  # Diccionario: {img_path: [indices]}
            print(f"Se cargó avance temporal desde {temp_features_file}.")
        except EOFError:
            pooled_vectors = []
            image_assignments = {}
            print(f"El archivo temporal {temp_features_file} está vacío. Se iniciará la extracción.")
    else:
        pooled_vectors = []
        image_assignments = {}
        print(f"No se encontró avance temporal; se iniciará la extracción de pooled features.")

    K = total_images  # Número total de imágenes de entrenamiento

    new_embeddings_count = 0

    # Extraer pooled features de las imágenes que aún no han sido procesadas
    for img_path in tqdm(image_files, desc="Extrayendo features", unit="imagen"):
        if img_path in image_assignments:
            continue  # Saltar imágenes ya procesadas
        img = pipeline.load_image(img_path)
        if img is None:
            image_assignments[img_path] = []  # Marca como procesada sin resultados
            continue
        preprocessed = pipeline.preprocess_image(img)
        pooled_features = pipeline.region_extractor.extract(preprocessed)
        if pooled_features.size == 0:
            image_assignments[img_path] = []
            continue
        indices_img = []
        # Se asume que pooled_features tiene forma (D, M)
        M = pooled_features.shape[1]  # Número de pooled vectors de la imagen
        for i in range(M):
            vec = pooled_features[:, i]
            pooled_vectors.append(vec)
            indices_img.append(len(pooled_vectors) - 1)
        image_assignments[img_path] = indices_img
        new_embeddings_count += 1
        # Guardar el avance cada 100 imágenes nuevas
        if new_embeddings_count % 100 == 0:
            temp_data = {"pooled_vectors": pooled_vectors, "image_assignments": image_assignments}
            with open(temp_features_file, 'wb') as f:
                pickle.dump(temp_data, f)

    if len(pooled_vectors) == 0:
        raise ValueError("No se extrajeron pooled features de ninguna imagen.")
    
    pooled_vectors = np.array(pooled_vectors)  # Forma: (total_vectors, D)
    print("Pooled vectors shape:", pooled_vectors.shape)

    # Entrenar clustering en el espacio de características
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(pooled_vectors)

    # Calcular la presencia de cada visual word: contar, para cada imagen, los clusters presentes.
    cluster_presence = np.zeros(num_clusters)
    for indices_img in image_assignments.values():
        if len(indices_img) == 0:
            continue
        img_vectors = pooled_vectors[indices_img]
        clusters = kmeans.predict(img_vectors)
        unique_clusters = np.unique(clusters)
        cluster_presence[unique_clusters] += 1

    # Calcular idf para cada visual word: Wc = log(K / nc)
    epsilon = 1e-6
    idf = np.log(K / (cluster_presence + epsilon))

    vocab = {"kmeans": kmeans, "idf": idf}

    # Guardar el vocabulario en el archivo especificado
    with open(vocabulary_file, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Vocabulario construido y guardado en {vocabulary_file}.")

    # Eliminar el archivo temporal (opcional)
    if os.path.exists(temp_features_file):
        os.remove(temp_features_file)

    return vocab

def explore_vocabulary(vocab_path):
    # Cargar el vocabulario desde el archivo pickle.
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    print("Claves en el vocabulario:", vocab.keys())
    
    # Extraer los elementos del vocabulario.
    kmeans = vocab.get("kmeans")
    idf = vocab.get("idf")
    
    if kmeans is not None:
        print("\nInformación del modelo k-means:")
        print("  Número de clusters (visual words):", kmeans.n_clusters)
        print("  Forma de los centros de cluster:", kmeans.cluster_centers_.shape)
        # Imprimir el primer centro como ejemplo.
        print("  Primer centro de cluster (sample):", kmeans.cluster_centers_[0])
    else:
        print("No se encontró la clave 'kmeans' en el vocabulario.")
    
    if idf is not None:
        print("\nInformación del vector idf:")
        print("  Forma del vector idf:", idf.shape)
        print("  Estadísticas del idf:")
        print("    Mínimo:", np.min(idf))
        print("    Máximo:", np.max(idf))
        print("    Promedio:", np.mean(idf))
    else:
        print("No se encontró la clave 'idf' en el vocabulario.")
    
    return vocab