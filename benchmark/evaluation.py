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
import matplotlib as mpl

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
    ks = [1, 5, 10]
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

    # Easy
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g["ok"] = np.concatenate([gnd[i]["easy"]])
        g["junk"] = np.concatenate([gnd[i]["junk"], gnd[i]["hard"]])
        gnd_t.append(g)
    mapE, apsE, mprE, prsE = compute_map(final_ranks, gnd_t, ks)

    # Medium
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g["ok"] = np.concatenate([gnd[i]["easy"], gnd[i]["hard"]])
        g["junk"] = np.concatenate([gnd[i]["junk"]])
        gnd_t.append(g)
    mapM, apsM, mprM, prsM = compute_map(final_ranks, gnd_t, ks)

    # Hard
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

def evaluate_search(
    df_result,
    places_db,
    place_of_img,
    q_place_gt,
    dataset: str = "rparis6k",
    use_bbox: bool = False,
    k_max = 30
):
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

    final_ranks = np.argsort(-sims_img, axis=0).T
    place_mat = ranks_to_places(final_ranks, place_of_img)    
    result = grid_by_place_with_gt(place_mat, q_place_gt, k_max)

    return result, final_ranks, place_mat, sims_img

def build_place_of_img_from_cfg(cfg):
    """
    db_img_id -> place_id compacto (0..P-1) o -1 si sin lugar.
    """
    db_image_names = np.array(cfg['imlist']) + cfg['ext']
    n_db = len(db_image_names)
    img2place = cfg['image_to_place']

    place_to_id = {}
    id_to_place = []
    next_pid = 0

    place_of_img = np.full(n_db, -1, dtype=np.int32)

    for db_id, fname in enumerate(db_image_names):
        pstr = img2place.get(fname, None)
        if pstr is None:
            continue
        if pstr not in place_to_id:
            place_to_id[pstr] = next_pid
            id_to_place.append(pstr)
            next_pid += 1
        place_of_img[db_id] = place_to_id[pstr]

    return place_of_img, place_to_id, id_to_place

def build_q_place_gt(cfg, place_to_id, place_of_img):
    """
    q_place_gt (Q,) con el place_id correcto por query.
    Reutiliza place_to_id; si falta el mapping para la query,
    infiere el place desde sus positivos (easy ∪ hard).
    """
    qim_names = np.array(cfg['qimlist']) + cfg['ext']
    img2place = cfg['image_to_place']
    gnd = cfg['gnd']

    q_place_gt = np.full(len(qim_names), -1, dtype=np.int32)

    for qi, qname in enumerate(qim_names):
        pstr = img2place.get(qname, None)
        if (pstr is not None) and (pstr in place_to_id):
            q_place_gt[qi] = place_to_id[pstr]
            continue

        # Fallback: inferir a partir de positivos (easy ∪ hard)
        pos = np.concatenate([gnd[qi].get('easy', []), gnd[qi].get('hard', [])])
        if pos.size == 0:
            continue

        pids = place_of_img[pos]
        pids = pids[pids >= 0]
        if pids.size == 0:
            continue

        vals, cnts = np.unique(pids, return_counts=True)
        q_place_gt[qi] = vals[np.argmax(cnts)]

    return q_place_gt

def ranks_to_places(final_ranks: np.ndarray, place_of_img: np.ndarray) -> np.ndarray:
    """
    final_ranks: (Q, N) con db_img_id por celda.
    place_of_img: (n_db,) con place_id por db_img_id (-1 si sin lugar).
    """
    return place_of_img[final_ranks]

def grid_by_place_with_gt(place_mat, q_place_gt, k_max=100):
    """
    place_mat: (Q, N) con place_id por celda; -1 = sin lugar
    q_place_gt: (Q,) place_id correcto de cada query; -1 = desconocido (se excluye de las métricas)
    """
    Q, N = place_mat.shape
    k_max = min(k_max, N)

    valid_q = (q_place_gt >= 0)
    Qv = int(valid_q.sum())
    if Qv == 0:
        raise ValueError("No hay queries con ground truth de place válido.")

    acc   = np.zeros((k_max, k_max), dtype=np.float32)
    cov   = np.zeros((k_max, k_max), dtype=np.float32)
    precd = np.zeros((k_max, k_max), dtype=np.float32)
    valid = np.zeros((k_max, k_max), dtype=bool)

    for k in range(1, k_max+1):
        topk = place_mat[:, :k]
        for v in range(1, k+1):
            decided = 0
            correct = 0
            # recorremos solo queries con GT válido
            for i in np.flatnonzero(valid_q):
                vals = topk[i]
                vals = vals[vals >= 0]
                if vals.size == 0:
                    continue
                counts = np.bincount(vals)
                maxc = counts.max(initial=0)
                if maxc >= v:
                    decided += 1
                    # place predicho = argmax conteo (desempate por id más chico)
                    pred = counts.argmax()
                    if pred == q_place_gt[i]:
                        correct += 1
            cov[k-1, v-1] = decided / Qv
            acc[k-1, v-1] = correct / Qv
            precd[k-1, v-1] = (correct / decided) if decided > 0 else 0.0
            valid[k-1, v-1] = True

    return {"accuracy": acc, "coverage": cov, "precision_decided": precd, "valid": valid}

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

# ============================
# Heatmap genérico
# ============================
def _plot_heatmap(Z: np.ndarray, valid: np.ndarray, title: str, vmin=0.0, vmax=1.0, cmap="viridis"):
    """
    Z: (k_max, k_max) indexado [k-1, v-1]
    valid: máscara booleana (v<=k)
    """
    Zmask = np.ma.array(Z, mask=~valid)
    cmap_obj = mpl.cm.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#cccccc")  # inválidos en gris claro
    im = plt.imshow(Zmask.T, origin="lower", aspect="auto",
                    extent=[1, Z.shape[0], 1, Z.shape[1]], vmin=vmin, vmax=vmax, cmap=cmap_obj)
    plt.xlabel("top_k")
    plt.ylabel("min_votes")
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)

def plot_heatmaps(acc: np.ndarray, cov: np.ndarray, precd: np.ndarray, valid: np.ndarray,
                  lambda_k: float = 0.0, lambda_v: float = 0.0):
    """
    Dibuja 3 heatmaps (accuracy, coverage, precision_decided).
    """
    plt.figure(figsize=(10, 4))

    # Coverage
    plt.subplot(1, 3, 2)
    _plot_heatmap(cov, valid, "Coverage")

    # Precision (solo entre decididas)
    plt.subplot(1, 3, 3)
    _plot_heatmap(precd, valid, "Precision (decididas)")

    plt.tight_layout()
    plt.show()

# ============================
# Cortes (curvas) para análisis fino
# ============================
def plot_slices_vs_min_votes(acc: np.ndarray, cov: np.ndarray, precd: np.ndarray, valid: np.ndarray,
                             k_list=(5, 10, 20, 50)):
    """
    Curvas (accuracy, coverage, precision_decided) vs min_votes para k fijos.
    """
    k_max = acc.shape[0]
    plt.figure(figsize=(11, 4))

    # Coverage
    plt.subplot(1, 3, 2)
    for k in k_list:
        if 1 <= k <= k_max:
            vmask = valid[k-1, :k]
            plt.plot(np.arange(1, k+1)[vmask], cov[k-1, :k][vmask], label=f"k={k}", marker="o", lw=1.5, ms=4)
    plt.xlabel("min_votes", fontsize=10)
    plt.ylabel("Cobertura", fontsize=10)
    # plt.title("Coverage vs min_votes")
    plt.legend(fontsize=8, title_fontsize=9)
    plt.grid(True)
    plt.tick_params(axis="both", which="major", labelsize=8)

    # Precision entre decididas
    plt.subplot(1, 3, 3)
    for k in k_list:
        if 1 <= k <= k_max:
            vmask = valid[k-1, :k]
            plt.plot(np.arange(1, k+1)[vmask], precd[k-1, :k][vmask], label=f"k={k}", marker="o", lw=1.5, ms=4)
    plt.xlabel("min_votes", fontsize=10)
    plt.ylabel("Precisión (decididas)", fontsize=10)
    # plt.title("Precision (decididas) vs min_votes")
    # plt.legend()
    plt.grid(True)
    plt.tick_params(axis="both", which="major", labelsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    plt.savefig('slices_vs_min_votes_graph_pa.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_slices_vs_topk(acc: np.ndarray, cov: np.ndarray, precd: np.ndarray, valid: np.ndarray,
                        v_list=(1, 2, 3, 5)):
    """
    Curvas (accuracy, coverage, precision_decided) vs top_k para min_votes fijos.
    """
    k_max = acc.shape[0]
    plt.figure(figsize=(11, 4))

    # Coverage
    plt.subplot(1, 3, 2)
    for v in v_list:
        ks = np.arange(max(v,1), k_max+1)
        mask = np.array([valid[k-1, v-1] for k in ks])
        plt.plot(ks[mask], np.array([cov[k-1, v-1] for k in ks])[mask], label=f"min_votes={v}", marker="o", lw=1.5, ms=4)
    plt.xlabel("top_k", fontsize=10)
    plt.ylabel("Cobertura", fontsize=10)
    # plt.title("Coverage vs top_k")
    plt.legend(fontsize=8, title_fontsize=9)
    plt.grid(True)
    plt.tick_params(axis="both", which="major", labelsize=8)

    # Precision entre decididas
    plt.subplot(1, 3, 3)
    for v in v_list:
        ks = np.arange(max(v,1), k_max+1)
        mask = np.array([valid[k-1, v-1] for k in ks])
        plt.plot(ks[mask], np.array([precd[k-1, v-1] for k in ks])[mask], label=f"min_votes={v}", marker="o", lw=1.5, ms=4)
    plt.xlabel("top_k", fontsize=10)
    plt.ylabel("Precisión (decididas)", fontsize=10)
    # plt.title("Precision (decididas) vs top_k")
    # plt.legend()

    plt.tick_params(axis="both", which="major", labelsize=8)

    plt.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    plt.savefig('slices_vs_topk_graph_pa.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_coverage_vs_precision(
    acc: np.ndarray,          # (K,K)
    cov: np.ndarray,          # (K,K)
    precd: np.ndarray,        # (K,K)
    valid: np.ndarray,        # (K,K) bool
    k_list=None,              # iterable de k a mostrar (por defecto: todos)
    label_every=0,            # 0 = sin rótulos de v; si >0, rotula cada 'label_every'
    zoom=None,                # None o (xmin,xmax,ymin,ymax) para recortar ejes
    figsize=(7,5),
    title="",
):
    """
    Un gráfico: eje X = Cobertura, eje Y = Precisión (entre decididas).
    - Una curva por k.
    - Los puntos de la curva son min_votes = 1..k.
    - Opcional: rótulos de v espaciados y ventana de zoom fija.
    """
    K = acc.shape[0]
    if k_list is None:
        k_list = range(1, K+1)

    fig, ax = plt.subplots(figsize=(5, 4))  # Tamaño proporcional a cada subplot anterior

    cmap = plt.cm.get_cmap("tab10", len(list(k_list)))

    for idx, k in enumerate(k_list):
        if not (1 <= k <= K):
            continue
        vs = np.arange(1, k+1)
        m  = valid[k-1, vs-1]
        if not np.any(m):
            continue

        xs = cov[k-1, vs-1][m]       # cobertura
        ys = precd[k-1, vs-1][m]     # precisión condicional
        ax.plot(xs, ys, marker="o", lw=1.5, ms=4, color=cmap(idx), label=f"k={k}")

        # Rótulos de v opcionales, espaciados
        if label_every and xs.size:
            for j, v in enumerate(vs[m]):
                if j % label_every == 0:
                    ax.text(xs[j], ys[j], f"v={v}", fontsize=7,
                            ha="left", va="bottom", color=cmap(idx))

    # Etiquetas de ejes y título con tamaño de fuente explícito
    ax.set_xlabel("Cobertura", fontsize=10)
    ax.set_ylabel("Precisión (decididas)", fontsize=10)
    ax.set_title(title, fontsize=11)

    # Ajuste de ticks
    ax.tick_params(axis="both", which="major", labelsize=8)

    # Leyenda con tamaño de fuente explícito
    ax.legend(title="top_k", frameon=True, fontsize=8, title_fontsize=9)

    ax.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    plt.savefig('coverage_vs_precision_graph_pa.png', dpi=300, bbox_inches='tight')

    plt.show()

def grid_min_sim_with_gt(
    place_mat: np.ndarray,     # (Q, N) place_id por celda; -1 = sin lugar
    score_mat: np.ndarray,     # (Q, N) similitud alineada con place_mat
    q_place_gt: np.ndarray,    # (Q,) place_id correcto por query; -1 = sin GT
    topk: int,
    min_votes: int,
    sim_thresholds: np.ndarray # 1D, p.ej. np.linspace(0.0, 1.0, 51)
):
    """
    Calcula accuracy, coverage y precision_decided al variar min_sim,
    con (topk, min_votes) fijos.
    """
    Q, N = place_mat.shape
    assert score_mat.shape == (Q, N), "score_mat debe estar alineada con place_mat"
    topk = min(topk, N)
    if not (1 <= min_votes <= topk):
        raise ValueError("Debe cumplirse 1 <= min_votes <= topk.")

    valid_q = (q_place_gt >= 0)
    Qv = int(valid_q.sum())
    if Qv == 0:
        raise ValueError("No hay queries con ground truth de place válido.")

    A = np.zeros(len(sim_thresholds), dtype=np.float32)  # accuracy
    C = np.zeros(len(sim_thresholds), dtype=np.float32)  # coverage (decididas/GT)
    P = np.zeros(len(sim_thresholds), dtype=np.float32)  # precisión entre decididas

    pk = place_mat[:, :topk]
    sk = score_mat[:, :topk]

    for idx, thr in enumerate(sim_thresholds):
        decided = 0
        correct = 0
        for i in np.flatnonzero(valid_q):
            mask = sk[i] >= thr
            if not np.any(mask):
                continue
            vals = pk[i, mask]
            vals = vals[vals >= 0]
            if vals.size == 0:
                continue
            cnts = np.bincount(vals)
            if cnts.max(initial=0) >= min_votes:
                decided += 1
                pred = cnts.argmax()
                if pred == q_place_gt[i]:
                    correct += 1
        C[idx] = decided / Qv
        A[idx] = correct / Qv
        P[idx] = (correct / decided) if decided > 0 else 0.0

    return {
        "min_sim": np.array(sim_thresholds, dtype=float),
        "accuracy": A,
        "coverage": C,
        "precision_decided": P,
        "topk": int(topk),
        "min_votes": int(min_votes),
    }

def build_score_mat(final_ranks: np.ndarray, sims_img: np.ndarray) -> np.ndarray:
    """
    final_ranks: (Q, N) con db_img_id por celda
    sims_img   : (n_db, Q) similitud por (db, query)
    Retorna score_mat (Q, N) tal que score_mat[q, j] = sims_img[final_ranks[q, j], q]
    """
    Q, N = final_ranks.shape
    score_mat = np.empty((Q, N), dtype=np.float32)
    for q in range(Q):
        score_mat[q] = sims_img[final_ranks[q], q]
    return score_mat

def plot_cov_prec_vs_min_sim(res, zoom_range=(0.8, 0.88), zoom_range_y=(0.0, 1.0)):
    """
    Muestra Cobertura y Precisión vs min_sim en una grilla 2x2.
    - Fila superior: ejes completos
    - Fila inferior: zoom en el rango de interés (zoom_range)
    """
    x = res["min_sim"]
    y_cov = res["coverage"]
    y_prec = res["precision_decided"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Cobertura - completo
    axes[0, 0].plot(x, y_cov, marker="o", lw=1.6, ms=4, color="tab:blue")
    axes[0, 0].set_xlabel("min_sim", fontsize=10)
    axes[0, 0].set_ylabel("Cobertura", fontsize=10)
    axes[0, 0].grid(True)
    axes[0, 0].tick_params(axis="both", which="major", labelsize=8)

    # Cobertura - zoom
    axes[1, 0].plot(x, y_cov, marker="o", lw=1.6, ms=4, color="tab:blue")
    axes[1, 0].set_xlabel("min_sim", fontsize=10)
    axes[1, 0].set_ylabel("Cobertura", fontsize=10)
    axes[1, 0].grid(True)
    axes[1, 0].set_xlim(*zoom_range)
    axes[1, 0].set_ylim(*zoom_range_y)
    axes[1, 0].tick_params(axis="both", which="major", labelsize=8)

    # Precisión - completo
    axes[0, 1].plot(x, y_prec, marker="o", lw=1.6, ms=4, color="tab:green")
    axes[0, 1].set_xlabel("min_sim")
    axes[0, 1].set_ylabel("Precisión (decididas)")
    axes[0, 1].grid(True)
    axes[0, 1].tick_params(axis="both", which="major", labelsize=8)

    # Precisión - zoom
    axes[1, 1].plot(x, y_prec, marker="o", lw=1.6, ms=4, color="tab:green")
    axes[1, 1].set_xlabel("min_sim")
    axes[1, 1].set_ylabel("Precisión (decididas)")
    axes[1, 1].grid(True)
    axes[1, 1].set_xlim(*zoom_range)
    axes[1, 1].set_ylim(*zoom_range_y)
    axes[1, 1].tick_params(axis="both", which="major", labelsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    plt.savefig('cov_prec_vs_min_sim_graph_pa.png', dpi=300, bbox_inches='tight')
    plt.show()
