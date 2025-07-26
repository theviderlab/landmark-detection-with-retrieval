import os
import json
from benchmark.revisitop.dataset import configdataset
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from benchmark.revisitop.evaluate import compute_map
from landmark_detection.search import Similarity_Search
import pandas as pd

def run_evaluation(
    df_result,
    descriptors_final,
    dataset: str = "rparis6k",   # Set test dataset: roxford5k | rparis6k
    use_bbox: bool = False,
):
    """Evaluates retrieval performance over the Revisited Oxford/Paris datasets.

    Parameters
    ----------
    df_result : pandas.DataFrame
        Tabla con los resultados obtenidos durante la fase de búsqueda. Debe
        contener, al menos, las columnas ``image_name`` y ``class_id``.
    descriptors_final : numpy.ndarray
        Descriptores de todas las imágenes en el mismo orden que ``df_result``.
    dataset : str, optional
        Nombre del *dataset* a evaluar (``roxford5k`` o ``rparis6k``).
    use_bbox : bool, optional
        Si es ``True`` no se filtran las filas con ``class_id`` distinto de -1,
        permitiendo que las similitudes se calculen utilizando también las
        detecciones en forma de ``bounding boxes``.

    Returns
    -------
    dict
        Métricas de rendimiento calculadas para los distintos niveles de
        dificultad.
    """

    DATASETS_PATH = os.path.abspath("datasets")

    #---------------------------------------------------------------------
    # Evaluate
    #---------------------------------------------------------------------

    print('>> {}: Evaluating test dataset...'.format(dataset)) 
    # config file for the dataset
    # separates query image list from database image list, when revisited protocol used
    cfg = configdataset(dataset, DATASETS_PATH)


    # ------------------------------------------------------------------
    # Asociamos cada nombre de imagen con su identificador de query y de
    # base de datos. Las imágenes que no pertenecen a ninguno de los grupos
    # reciben el identificador -1.
    # ------------------------------------------------------------------
    query_image_names = np.array(cfg['qimlist']) + cfg['ext']
    q_idx_map = {name: idx for idx, name in enumerate(query_image_names)}
    df_result['q_img_id'] = np.nan
    df_result.loc[:, 'q_img_id'] = df_result.loc[:, 'image_name'].map(q_idx_map)
    df_result['q_img_id'] = df_result['q_img_id'].fillna(-1).astype(int)

    # Extract database results
    db_image_names = np.array(cfg['imlist']) + cfg['ext']
    db_idx_map = {name: idx for idx, name in enumerate(db_image_names)}
    df_result['db_img_id'] = np.nan
    df_result.loc[:, 'db_img_id'] = df_result.loc[:, 'image_name'].map(db_idx_map)
    df_result['db_img_id'] = df_result['db_img_id'].fillna(-1).astype(int)

    # --------------------------------------------------------------
    # Carga de descriptores de consultas y de base de datos
    # --------------------------------------------------------------
    print('>> {}: Loading features...'.format(dataset))

    mask_img_full = df_result['class_id'] == -1
    mask_selection = np.ones(len(df_result), dtype=bool) if use_bbox else mask_img_full

    # Extraemos descriptores de las imágenes de consulta en el orden
    # definido por el dataset para asegurar correspondencia con ``gnd``.
    mask_query = df_result['image_name'].isin(query_image_names)
    query_index = (
        df_result[mask_query & mask_img_full]
        .sort_values("q_img_id")
        .index
    )
    Q = descriptors_final[query_index]

    # Extraemos descriptores de la base de datos. El orden no es
    # relevante ya que posteriormente transformaremos los índices a
    # identificadores del dataset.
    mask_db = df_result['image_name'].isin(db_image_names)
    db_index = (
        df_result[mask_db & mask_selection]
        .sort_values("db_img_id")
        .index
    )
    X = descriptors_final[db_index]

    # --------------------------------------------------------------
    # Búsqueda por similitud usando producto escalar (coseno si las
    # características están normalizadas). ``ranks`` contiene para cada
    # consulta el índice de las imágenes de base de datos ordenadas por
    # relevancia.
    # --------------------------------------------------------------
    print('>> {}: Retrieval...'.format(dataset))
    searcher = Similarity_Search()
    ranks = searcher.compute_ranks(Q, X).cpu().numpy()

    # --------------------------------------------------------------
    # Transformamos los índices obtenidos en ``ranks`` para que hagan
    # referencia al identificador real dentro del dataset revisitado.
    # --------------------------------------------------------------
    ranks_id = db_index.values[ranks]
    final_ranks = df_result['db_img_id'].values[ranks_id]

    if use_bbox:
        n_db = len(db_image_names)
        unique_ranks = np.zeros((n_db, final_ranks.shape[1]), dtype=int)
        all_ids = np.arange(n_db)
        for i in range(final_ranks.shape[1]):
            seen = set()
            uniq = []
            for idx in final_ranks[:, i]:
                if idx not in seen and idx != -1:
                    seen.add(idx)
                    uniq.append(idx)
                if len(seen) == n_db:
                    break
            if len(seen) < n_db:
                remaining = [id_ for id_ in all_ids if id_ not in seen]
                uniq.extend(remaining)
            unique_ranks[:, i] = np.array(uniq[:n_db])
        final_ranks = unique_ranks

    # revisited evaluation
    gnd = cfg['gnd']

    # evaluate ranks
    ks = [1, 5, 10]

    # search for easy
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['easy']])
        g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
        gnd_t.append(g)
    mapE, apsE, mprE, prsE = compute_map(final_ranks, gnd_t, ks)

    # search for easy & hard
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
        g['junk'] = np.concatenate([gnd[i]['junk']])
        gnd_t.append(g)
    mapM, apsM, mprM, prsM = compute_map(final_ranks, gnd_t, ks)

    # search for hard
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['hard']])
        g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
        gnd_t.append(g)
    mapH, apsH, mprH, prsH = compute_map(final_ranks, gnd_t, ks)

    print(
        '>> {}: mAP E: {}, M: {}, H: {}'.format(
            dataset,
            np.around(mapE * 100, decimals=2),
            np.around(mapM * 100, decimals=2),
            np.around(mapH * 100, decimals=2),
        )
    )
    print(
        '>> {}: mP@k{} E: {}, M: {}, H: {}'.format(
            dataset,
            np.array(ks),
            np.around(mprE * 100, decimals=2),
            np.around(mprM * 100, decimals=2),
            np.around(mprH * 100, decimals=2),
        )
    )

    return {
        "map_easy": mapE,
        "map_medium": mapM,
        "map_hard": mapH,
        "mpr_easy": mprE,
        "mpr_medium": mprM,
        "mpr_hard": mprH,
    }

def run_evaluation2(
    df_result,
    places_db,
    dataset: str = "rparis6k",
    use_bbox: bool = False,
):
    """Alternative evaluation using bounding boxes on queries.

    This function expects ``places_db`` with descriptors concatenated with
    an ``image_id`` in the last column. When ``use_bbox`` is ``True`` both
    query and database detections are considered during retrieval.
    """

    DATASETS_PATH = os.path.abspath("datasets")

    print(f">> {dataset}: Evaluating test dataset...")
    cfg = configdataset(dataset, DATASETS_PATH)

    query_image_names = np.array(cfg["qimlist"]) + cfg["ext"]
    q_idx_map = {name: idx for idx, name in enumerate(query_image_names)}
    df_result["q_img_id"] = df_result["image_name"].map(q_idx_map).fillna(-1).astype(int)

    db_image_names = np.array(cfg["imlist"]) + cfg["ext"]
    db_idx_map = {name: idx for idx, name in enumerate(db_image_names)}
    df_result["db_img_id"] = df_result["image_name"].map(db_idx_map).fillna(-1).astype(int)

    print(f">> {dataset}: Loading features...")

    mask_img_full = df_result["class_id"] == -1
    mask_selection = np.ones(len(df_result), dtype=bool) if use_bbox else mask_img_full

    mask_query = df_result["image_name"].isin(query_image_names)
    query_index = (
        df_result[mask_query & mask_selection]
        .sort_values("q_img_id")
        .index
    )
    Q = places_db[query_index, :-1]
    q_ids = df_result.loc[query_index, "q_img_id"].values

    mask_db = df_result["image_name"].isin(db_image_names)
    db_index = (
        df_result[mask_db & mask_selection]
        .sort_values("db_img_id")
        .index
    )
    X = places_db[db_index, :-1]
    db_ids = df_result.loc[db_index, "db_img_id"].values

    print(f">> {dataset}: Retrieval...")
    sims_desc = np.dot(Q, X.T)

    n_db = len(db_image_names)
    n_q = len(query_image_names)
    sims_img = np.full((n_db, n_q), -np.inf, dtype=np.float32)

    for qid in range(n_q):
        rows = np.where(q_ids == qid)[0]
        if rows.size == 0:
            continue
        s_q = sims_desc[rows]
        for dbid in range(n_db):
            cols = np.where(db_ids == dbid)[0]
            if cols.size == 0:
                continue
            sims_img[dbid, qid] = np.max(s_q[:, cols])

    final_ranks = np.argsort(-sims_img, axis=0)

    gnd = cfg["gnd"]
    ks = [1, 5, 10]

    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g["ok"] = np.concatenate([gnd[i]["easy"]])
        g["junk"] = np.concatenate([gnd[i]["junk"], gnd[i]["hard"]])
        gnd_t.append(g)
    mapE, apsE, mprE, prsE = compute_map(final_ranks, gnd_t, ks)

    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g["ok"] = np.concatenate([gnd[i]["easy"], gnd[i]["hard"]])
        g["junk"] = np.concatenate([gnd[i]["junk"]])
        gnd_t.append(g)
    mapM, apsM, mprM, prsM = compute_map(final_ranks, gnd_t, ks)

    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g["ok"] = np.concatenate([gnd[i]["hard"]])
        g["junk"] = np.concatenate([gnd[i]["junk"], gnd[i]["easy"]])
        gnd_t.append(g)
    mapH, apsH, mprH, prsH = compute_map(final_ranks, gnd_t, ks)

    print(
        '>> {}: mAP E: {}, M: {}, H: {}'.format(
            dataset,
            np.around(mapE * 100, decimals=2),
            np.around(mapM * 100, decimals=2),
            np.around(mapH * 100, decimals=2),
        )
    )
    print(
        '>> {}: mP@k{} E: {}, M: {}, H: {}'.format(
            dataset,
            np.array(ks),
            np.around(mprE * 100, decimals=2),
            np.around(mprM * 100, decimals=2),
            np.around(mprH * 100, decimals=2),
        )
    )

    return {
        "map_easy": mapE,
        "map_medium": mapM,
        "map_hard": mapH,
        "mpr_easy": mprE,
        "mpr_medium": mprM,
        "mpr_hard": mprH,
    }

def show_inference_example(
    df_result,
    descriptors_final,
    q_idx=0,
    top_n: int = 5,
    dataset: str = "rparis6k",
    use_bbox: bool = False,
):
    """Muestra un ejemplo de inferencia para una o varias consultas.

    Parameters
    ----------
    df_result : pandas.DataFrame
        Tabla con los resultados de detección y descriptores.
    descriptors_final : numpy.ndarray
        Descriptores de las imágenes en el mismo orden que ``df_result``.
    q_idx : int or Sequence[int], optional
        Índice o índices de las imágenes de consulta a visualizar. Cada
        consulta se mostrará en una fila.
    top_n : int, optional
        Número de imágenes recuperadas a mostrar.
    dataset : str, optional
        Conjunto de evaluación (``roxford5k`` o ``rparis6k``).
    use_bbox : bool, optional
        Si es ``True`` se emplean las ``bounding boxes`` para calcular
        similitud.
    """

    DATASETS_PATH = os.path.abspath("datasets")

    cfg = configdataset(dataset, DATASETS_PATH)

    query_image_names = np.array(cfg["qimlist"]) + cfg["ext"]
    q_idx_map = {name: idx for idx, name in enumerate(query_image_names)}
    df = df_result.copy()
    df["q_img_id"] = df["image_name"].map(q_idx_map).fillna(-1).astype(int)

    db_image_names = np.array(cfg["imlist"]) + cfg["ext"]
    db_idx_map = {name: idx for idx, name in enumerate(db_image_names)}
    df["db_img_id"] = df["image_name"].map(db_idx_map).fillna(-1).astype(int)

    mask_img_full = df["class_id"] == -1
    mask_selection = np.ones(len(df), dtype=bool) if use_bbox else mask_img_full

    mask_query = df["image_name"].isin(query_image_names)
    query_index = (
        df[mask_query & mask_img_full].sort_values("q_img_id").index
    )
    Q = descriptors_final[query_index]

    mask_db = df["image_name"].isin(db_image_names)
    db_index = (
        df[mask_db & mask_selection].sort_values("db_img_id").index
    )
    X = descriptors_final[db_index]

    if isinstance(q_idx, (list, tuple, np.ndarray)):
        q_indices = list(q_idx)
    else:
        q_indices = [q_idx]

    for qi in q_indices:
        if qi < 0 or qi >= len(Q):
            raise ValueError("El indice de consulta esta fuera de rango")

    n_rows = len(q_indices)
    fig, axes = plt.subplots(n_rows, top_n + 1, figsize=(3 * (top_n + 1), 3 * n_rows))
    axes = np.atleast_2d(axes)

    for row, qi in enumerate(q_indices):
        sim = np.dot(X, Q[qi])
        ranks = np.argsort(-sim)
        final_ranks = df.loc[db_index, "db_img_id"].values[ranks]

        gnd = cfg["gnd"][qi]
        ok_ids = set(np.concatenate([gnd.get("easy", []), gnd.get("hard", [])]))

        n_show = min(top_n, len(final_ranks))

        q_path = cfg["qim_fname"](cfg, qi)
        q_img = cv2.cvtColor(cv2.imread(q_path), cv2.COLOR_BGR2RGB)
        axes[row, 0].imshow(q_img)
        axes[row, 0].set_title("Query")
        axes[row, 0].axis("off")

        for i in range(n_show):
            db_id = final_ranks[i]
            img_path = cfg["im_fname"](cfg, db_id)
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            ax = axes[row, i + 1]
            ax.imshow(img)
            ax.axis("off")
            if db_id in ok_ids:
                color = "green"
                label = "correcto"
            else:
                color = "red"
                label = "incorrecto"
            rect = patches.Rectangle(
                (0, 0), img.shape[1], img.shape[0], linewidth=4, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
            ax.set_title(f"{i + 1}: {label}", color=color)

        for j in range(n_show, top_n):
            axes[row, j + 1].axis("off")

    plt.tight_layout()
    plt.show()

def save_evaluation_result(
    results: dict, path: str, config: dict, results_bbox: dict | None = None
) -> None:
    """Store evaluation metrics and configuration in a JSON file.

    This helper collects the results returned by :func:`run_evaluation` or
    :func:`run_evaluation2` together with the configuration used to obtain
    them.

    Parameters
    ----------
    results : dict
        Dictionary with metrics produced by the evaluation functions.
    path : str
        Destination JSON file where results will be stored.
    config : dict
        Configuration parameters relevant to the evaluation run.
    results_bbox : dict | None, optional
        Additional bounding box metrics to store alongside ``results``.
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    data = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []

    entry = {"results": results, "config": config}
    if results_bbox is not None:
        entry["results_bbox"] = results_bbox
    data.append(entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

