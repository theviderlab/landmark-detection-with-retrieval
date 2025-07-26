import os
import cv2
import yaml
import numpy as np
import json
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

def show_bboxes(
    img_path,
    boxes=None,
    class_names_path=None,
    cls=None,
    scores=None,
    bbox_gnd=None,
):
    """
    Muestra una imagen con las cajas predichas en rojo y opcionalmente la caja ground truth en verde.
    Si el índice de clase es -1, etiqueta la caja como "full image".

    Args:
        img_path (str): Ruta a la imagen.
        boxes (List[List[float]]): Lista de cajas predichas ``[[x1, y1, x2, y2], ...]``.
        class_names_path (Optional[str]): Ruta al YAML con los nombres de las clases.
        cls (Optional[List[int]]): Lista de índices de clase para cada caja predicha.
        scores (Optional[List[float]]): Lista de puntajes de confianza para cada caja predicha.
        bbox_gnd (Optional[List[float]]): Caja ground truth ``[x1, y1, x2, y2]``. Si no se
            provee, no la dibuja.

    Si ``class_names_path``, ``cls`` o ``scores`` son ``None`` se muestran las
    cajas sin etiquetas.
    """
    # Carga de imagen
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"No se encontró la imagen en {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Carga las clases si se proporciona un YAML
    class_names = []
    if class_names_path is not None:
        class_names = load_names_from_yaml(class_names_path)

    # Normalizar listas de entradas opcionales
    if cls is None:
        cls = [None] * len(boxes)

    if scores is None:
        scores = [None] * len(boxes)
    
    print(f"Encontradas {len(boxes)} cajas:")
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        idx = cls[i]
        if idx is None:
            class_name = "None"
        elif idx == -1:
            class_name = "No detected"
        else:
            class_name = class_names[idx] if idx < len(class_names) else str(idx)
        sc = scores[i]
        score_str = f"{sc:.2f}" if sc is not None else "None"
        print(
            f"  Clase {idx} {class_name} @ {score_str} → [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]"
        )
        
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
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2,
            edgecolor="red",
        )
        ax.add_patch(rect)
        label_parts = []
        if cls_idx is not None:
            if cls_idx == -1:
                label_parts.append("ND")
            else:
                label_parts.append(
                    class_names[cls_idx]
                    if cls_idx < len(class_names)
                    else str(cls_idx)
                )
        if sc is not None:
            label_parts.append(f"{sc:.2f}")
        if label_parts:
            ax.text(
                x1,
                y1 - 4,
                f"{i} - {':'.join(label_parts)}",
                color="white",
                fontsize=9,
                bbox=dict(facecolor="red", alpha=0.5),
            )
        i += 1
    
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
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    
    names_dict = data.get('names', {})
    # Ordenar los elementos por la clave (convirtiendo a entero si hace falta)
    sorted_items = sorted(names_dict.items(), key=lambda item: int(item[0]))
    
    # Construir la lista de valores
    names_list = [value for _, value in sorted_items]
    return names_list

def show_similarity_search(
    query_img_path,
    class_names_path,
    final_boxes,
    final_scores,
    final_classes,
):
    """Visualiza los resultados de :class:`Similarity_Search`.

    Parameters
    ----------
    final_boxes : Sequence[Sequence[float]]
        Cajas obtenidas tras la búsqueda de similitud.
    final_scores : Sequence[float]
        Puntuaciones de similitud de cada caja.
    final_classes : Sequence[int | None]
        ``landmark`` asignado a cada detección o ``None`` si no se pudo
        determinar.
    query_img_path : str
        Ruta de la imagen de consulta.
    class_names_path : str
        Archivo YAML con los nombres de las clases para mostrar en la figura.
    """

    final_boxes = np.asarray(final_boxes)
    final_scores = np.asarray(final_scores)

    q_img_bgr = cv2.imread(query_img_path)
    if q_img_bgr is None:
        raise FileNotFoundError(f"No se encontró la imagen en {query_img_path}")
    q_img_rgb = cv2.cvtColor(q_img_bgr, cv2.COLOR_BGR2RGB)

    class_names = load_names_from_yaml(class_names_path)

    n_boxes = final_boxes.shape[0]
    fig, axes = plt.subplots(1, n_boxes, figsize=(3 * n_boxes, 3))
    axes = np.atleast_1d(axes)

    for i in range(n_boxes):
        x1, y1, x2, y2 = map(int, final_boxes[i])
        crop = q_img_rgb[y1:y2, x1:x2]
        axes[i].imshow(crop)
        cls = final_classes[i]
        if cls is None:
            label = "None"
        elif 0 <= cls < len(class_names):
            label = class_names[cls]
        else:
            label = str(cls)
        axes[i].set_title(f"{label} ({final_scores[i]:.2f})")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

def export_places_db(places_db: np.ndarray, label_map: dict[int, str], output_dir: str) -> None:
    """Guarda la base de datos de descriptores y el mapeo de etiquetas.

    Parameters
    ----------
    places_db : numpy.ndarray
        Matriz ``(N, C + 1)`` con descriptores y el ``place_id`` en la última columna.
    label_map : dict[int, str]
        Diccionario que mapea ``place_id`` a su nombre legible.
    output_dir : str
        Carpeta donde se escribirán ``places_db.npz`` y ``label_map.json``.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(os.path.join(output_dir, "places_db.npz"), places_db.astype(np.float32))
    with open(os.path.join(output_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
