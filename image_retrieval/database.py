import os
import pickle
import numpy as np
from tqdm import tqdm

def build_database(pipeline, embeddings_file):
    """
    Construye la base de datos de embeddings para el pipeline a partir de las imágenes en el directorio.
    
    Para cada imagen en 'db_directory', la función verifica si ya existe el embedding en 'embeddings_file'.
    Si no existe, extrae las características usando el método 'extract_features' del pipeline.
    Guarda el avance cada 10 nuevos embeddings calculados.
    
    Args:
        pipeline: Instancia del pipeline que posee los métodos:
            - load_image(image_path)
            - preprocess_image(img)
            - extract_features(preprocessed_img)
          y atributos 'db_image_paths' y 'db_embeddings' que se actualizarán.
        embeddings_file (str): Ruta al archivo donde se guardarán los embeddings (diccionario).
    """
    # Cargar o inicializar el diccionario de embeddings.
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            embeddings_dict = pickle.load(f)
        print(f"Se cargaron embeddings existentes desde {embeddings_file}.")
    else:
        embeddings_dict = {}
        print(f"No se encontró el archivo de embeddings. Se creará uno nuevo: {embeddings_file}.")

    # Listar todas las imágenes en el directorio (considera formatos comunes).
    base_dir = os.path.dirname(os.path.abspath(__file__))
    abs_db_directory = os.path.normpath(os.path.join(base_dir, pipeline.db_directory))
    image_files = [os.path.join(abs_db_directory, file) for file in os.listdir(abs_db_directory)
                   if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_files)
    print(f"Encontradas {total_images} imágenes en {pipeline.db_directory}.")
    print("Construyendo la base de datos de embeddings...")

    computed_embeddings = []
    valid_paths = []
    new_embeddings_count = 0

    # Iterar sobre cada imagen con una barra de progreso.
    for idx, path in enumerate(tqdm(image_files, desc="Procesando imágenes", unit="imagen")):
        if path in embeddings_dict:
            embedding = embeddings_dict[path]
        else:
            img = pipeline.load_image(path)
            if img is None:
                continue
            preprocessed = pipeline.preprocess_image(img)
            embedding = pipeline.extract_features(preprocessed)
            embeddings_dict[path] = embedding
            new_embeddings_count += 1

            # Guardar el avance cada 10 nuevas extracciones.
            if new_embeddings_count % 10 == 0:
                with open(embeddings_file, 'wb') as f:
                    pickle.dump(embeddings_dict, f)
                # tqdm muestra el progreso, por lo que no es necesario imprimir un mensaje adicional.

        computed_embeddings.append(embedding)
        valid_paths.append(path)

    # Guardar el diccionario final de embeddings.
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings_dict, f)
    print(f"\nSe guardaron los embeddings en {embeddings_file}.")

    # Actualizar los atributos del pipeline.
    pipeline.db_embeddings = np.array(computed_embeddings)
    pipeline.db_image_paths = valid_paths
    print(f"Base de datos construida con {len(valid_paths)} imágenes.")
