import matplotlib.pyplot as plt
import cv2
import yaml

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
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False, linewidth=2, edgecolor='red'
        )
        ax.add_patch(rect)
        if cls_idx is None:
            class_name = "None"
        elif cls_idx == -1:
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