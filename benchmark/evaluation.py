import os
from benchmark.revisitop.dataset import configdataset
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from benchmark.revisitop.evaluate import compute_map

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
    sim = np.dot(X, Q.T)
    ranks = np.argsort(-sim, axis=0)

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

