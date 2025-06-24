import os
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt

def show_image(img_path):   
    # Carga de imagen
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"No se encontró la imagen en {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Mostrar la imagen original
    plt.figure(figsize=(6,6))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

def show_bboxes(img_path, class_names_path, boxes, cls, scores, bbox_gnd=None):
    """
    Muestra una imagen con las cajas predichas en rojo y opcionalmente la caja ground truth en verde.
    Si el índice de clase es -1, etiqueta la caja como "full image".

    Args:
        img_path (str): Ruta a la imagen.
        class_names_path (str): Ruta al YAML con los nombres de las clases.
        boxes (List[List[float]]): Lista de cajas predichas [[x1, y1, x2, y2], ...].
        cls (List[int]): Lista de índices de clase para cada caja predicha.
        scores (List[float]): Lista de puntajes de confianza para cada caja predicha.
        bbox_gnd (Optional[List[float]]): Caja ground truth [x1, y1, x2, y2]. Si no se provee, no la dibuja.
    """
    # Carga de imagen
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"No se encontró la imagen en {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Carga las clases
    class_names = load_names_from_yaml(class_names_path)
    
    print(f"Encontradas {len(boxes)} cajas:")
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        idx = cls[i]
        if idx is None:
            class_name = "None"
        elif idx == -1:
            class_name = "full image"
        else:
            class_name = class_names[idx]
        print(f"  Clase {idx} {class_name} @ {scores[i]:.2f} → [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")
        
    plt.figure(figsize=(6,6))
    plt.imshow(img_rgb)
    ax = plt.gca()

    # Dibujar caja ground truth en verde (si se proporcionó)
    if bbox_gnd is not None:
        gx1, gy1, gx2, gy2 = bbox_gnd
        rect_gnd = plt.Rectangle(
            (gx1, gy1),
            gx2 - gx1,
            gy2 - gy1,
            fill=False,
            linewidth=2,
            edgecolor='green'
        )
        ax.add_patch(rect_gnd)
        ax.text(
            gx1, gy1 - 4,
            "GT",
            color='white', fontsize=9,
            bbox=dict(facecolor='green', alpha=0.5)
        )
    
    # Dibujar cajas predichas en rojo
    i = 0
    for (x1, y1, x2, y2), cls_idx, sc in zip(boxes, cls, scores):
        if cls_idx is None:
            continue

        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False, linewidth=2, edgecolor='red'
        )
        ax.add_patch(rect)
        if cls_idx == -1:
            class_name = "full image"
        else:
            class_name = class_names[cls_idx]
        ax.text(
            x1, y1 - 4,
            f"{i} - {class_name}:{sc:.2f}",
            color='white', fontsize=9,
            bbox=dict(facecolor='red', alpha=0.5)
        )
        i = i + 1
    
    plt.axis('off')
    plt.show()

def load_names_from_yaml(file_path):
    """
    Carga un archivo YAML con la estructura:
    
    names:
      0: Accordion
      1: Adhesive tape
      2: Aircraft
      3: Airplane
      4: Alarm clock
      5: Alpaca
    
    y devuelve una lista de nombres ordenados según la clave numérica.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    names_dict = data.get('names', {})
    # Ordenar los elementos por la clave (convirtiendo a entero si hace falta)
    sorted_items = sorted(names_dict.items(), key=lambda item: int(item[0]))
    
    # Construir la lista de valores
    names_list = [value for _, value in sorted_items]
    return names_list


def show_similarity_search(
    results,
    top_sims,
    top_idx,
    df_result,
    landmark_tensor,
    final_boxes,
    query_img_path,
    image_folder,
    top_n: int = 5,
):
    """Visualiza los resultados de :class:`Similarity_Search`.

    Parameters
    ----------
    results : Sequence[int | None]
        Lista con el ``landmark`` asignado a cada detección de la consulta
        o ``None`` si la similitud mínima no se superó.
    top_sims : numpy.ndarray | torch.Tensor
        Matriz ``(D, K)`` con las similitudes de los ``K`` vecinos más
        cercanos para cada una de las ``D`` detecciones de la consulta.
    top_idx : numpy.ndarray | torch.Tensor
        Índices de ``df_result`` correspondientes a ``top_sims``.
    df_result : pandas.DataFrame
        Salida de :func:`benchmark.database.build_image_database` con la
        información de las detecciones de la base de datos.
    landmark_tensor : Sequence[int]
        Clase ``class_id`` asociada a cada fila de ``df_result``.
    final_boxes : list
        Resultado de :meth:`Pipeline_Yolo_CVNet_SG.run` utilizado para
        obtener las cajas de la imagen de consulta.
    query_img_path : str
        Ruta de la imagen de consulta.
    image_folder : str
        Carpeta donde se encuentran las imágenes referenciadas en
        ``df_result``.
    top_n : int, optional
        Número máximo de vecinos a visualizar por detección.
    """

    final_boxes = np.array(final_boxes)

    q_img_bgr = cv2.imread(query_img_path)
    if q_img_bgr is None:
        raise FileNotFoundError(f"No se encontró la imagen en {query_img_path}")
    q_img_rgb = cv2.cvtColor(q_img_bgr, cv2.COLOR_BGR2RGB)

    n_queries = final_boxes.shape[0]
    top_n = min(top_n, top_idx.shape[1])

    fig, axes = plt.subplots(n_queries, top_n + 1, figsize=(3 * (top_n + 1), 3 * n_queries))
    axes = np.atleast_2d(axes)

    for row in range(n_queries):
        x1, y1, x2, y2 = map(int, final_boxes[row])
        crop_q = q_img_rgb[y1:y2, x1:x2]
        axes[row, 0].imshow(crop_q)
        label = results[row] if row < len(results) else None
        axes[row, 0].set_title(f"Q{row}→{label}")
        axes[row, 0].axis("off")

        for col in range(top_n):
            db_id = int(top_idx[row, col])
            sim = float(top_sims[row, col])
            img_name = df_result.loc[db_id, "image_name"]
            bbox = df_result.loc[db_id, "bbox"]
            img_path = os.path.join(image_folder, img_name)
            db_img = cv2.imread(img_path)
            if db_img is None:
                raise FileNotFoundError(f"No se encontró la imagen en {img_path}")
            db_img = cv2.cvtColor(db_img, cv2.COLOR_BGR2RGB)
            x1d, y1d, x2d, y2d = map(int, bbox)
            crop_db = db_img[y1d:y2d, x1d:x2d]
            cls = int(landmark_tensor[db_id]) if landmark_tensor is not None else -1
            ax = axes[row, col + 1]
            ax.imshow(crop_db)
            ax.set_title(f"{cls} ({sim:.2f})")
            ax.axis("off")

        for col in range(top_n, axes.shape[1] - 1):
            axes[row, col + 1].axis("off")

    plt.tight_layout()
    plt.show()
