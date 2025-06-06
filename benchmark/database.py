"""Utilities to build detection descriptors database."""

from __future__ import annotations

import os
from typing import Tuple
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from landmark_detection.pipeline import Pipeline_Yolo_CVNet_SG

def build_image_database(
    pipeline: Pipeline_Yolo_CVNet_SG,
    image_folder: str,
    df_pickle_path: str,
    descriptor_pickle_path: str,
    force_rebuild: bool = False,
    save_every: int = 500,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Construye o actualiza una base de datos de detecciones y descriptores.

    Parameters
    ----------
    pipeline : Pipeline_Yolo_CVNet_SG
        Pipeline used to run detection and descriptor extraction.
    image_folder : str
        Directorio que contiene las imágenes a procesar.
    df_pickle_path : str
        Ruta al archivo pickle donde se guarda el DataFrame.
    descriptor_pickle_path : str
        Ruta al archivo pickle donde se guarda el array de descriptores.
    force_rebuild : bool, optional
        Si es True, se ignoran los archivos previos y se procesan todas las imágenes.
    save_every : int, optional
        Número de imágenes procesadas tras el cual se actualizan los pickles.

    Returns
    -------
    Tuple[pandas.DataFrame, numpy.ndarray]
        Updated dataframe and descriptor array.
    """
    # Validar la carpeta
    if not os.path.isdir(image_folder):
        raise FileNotFoundError(f"La ruta '{image_folder}' no es un directorio válido.")

    extensiones_validas = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    all_files = [
        f for f in os.listdir(image_folder)
        if os.path.splitext(f)[1].lower() in extensiones_validas
    ]

    columns = ["image_name", "bbox", "class_id", "confidence"]

    if not force_rebuild and os.path.isfile(df_pickle_path):
        df_result = pd.read_pickle(df_pickle_path)
        if not set(columns).issubset(df_result.columns):
            raise ValueError("El DataFrame cargado no contiene las columnas requeridas.")
    else:
        df_result = pd.DataFrame(columns=columns)

    if not force_rebuild and os.path.isfile(descriptor_pickle_path):
        with open(descriptor_pickle_path, "rb") as f:
            descriptors_final = pickle.load(f)
        if not isinstance(descriptors_final, np.ndarray):
            raise ValueError("El archivo de descriptores no contiene un numpy.ndarray.")
    else:
        descriptors_final = np.zeros((0, 0), dtype=np.float32)

    if not force_rebuild and len(df_result) != len(descriptors_final):
        raise ValueError("El número de filas en el DataFrame no coincide con el número de descriptores.")

    C = descriptors_final.shape[1] if descriptors_final.size else 0
    processed = set(df_result["image_name"].unique()) if not force_rebuild else set()
    images_to_process = [f for f in all_files if f not in processed]

    new_rows = []
    new_descriptors = []
    init_C = C == 0
    processed_since_save = 0

    for img_name in tqdm(images_to_process, desc="Procesando imágenes"):
        img_path = os.path.join(image_folder, img_name)
        try:
            final_boxes, final_scores, final_classes, descriptors = pipeline.run(img_path)
        except Exception as e:
            print(f"Error procesando {img_name}: {e}")
            continue

        boxes_np = final_boxes.numpy() if hasattr(final_boxes, "numpy") else np.asarray(final_boxes)
        scores_np = final_scores.numpy() if hasattr(final_scores, "numpy") else np.asarray(final_scores)
        classes_np = final_classes.numpy() if hasattr(final_classes, "numpy") else np.asarray(final_classes)
        descriptors_np = descriptors.numpy() if hasattr(descriptors, "numpy") else np.asarray(descriptors)

        if init_C:
            C = descriptors_np.shape[1]
            descriptors_final = np.zeros((0, C), dtype=descriptors_np.dtype)
            init_C = False

        for j in range(boxes_np.shape[0]):
            row = {
                "image_name": img_name,
                "bbox": tuple(map(float, boxes_np[j].tolist())),
                "class_id": int(classes_np[j].item()),
                "confidence": float(scores_np[j].item()),
            }
            new_rows.append(row)
            new_descriptors.append(descriptors_np[j].reshape(1, C))

        processed_since_save += 1
        if processed_since_save >= save_every:
            if new_rows:
                df_new = pd.DataFrame(new_rows)
                if df_result.empty:
                    df_result = df_new
                else:
                    df_result = pd.concat([df_result, df_new], ignore_index=True)
                stacked = np.vstack(new_descriptors)
                descriptors_final = np.vstack([descriptors_final, stacked]) if descriptors_final.size else stacked
                new_rows = []
                new_descriptors = []
                pd.to_pickle(df_result, df_pickle_path)
                with open(descriptor_pickle_path, "wb") as f:
                    pickle.dump(descriptors_final, f)
            processed_since_save = 0

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        if df_result.empty:
            df_result = df_new
        else:
            df_result = pd.concat([df_result, df_new], ignore_index=True)
        stacked = np.vstack(new_descriptors)
        descriptors_final = np.vstack([descriptors_final, stacked]) if descriptors_final.size else stacked

    pd.to_pickle(df_result, df_pickle_path)
    with open(descriptor_pickle_path, "wb") as f:
        pickle.dump(descriptors_final, f)

    return df_result, descriptors_final