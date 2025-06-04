from PIL import Image

def generate_scaled_images(img_pil, scales):
    """
    Genera múltiples versiones escaladas de una imagen PIL.

    Args:
        img_pil (PIL.Image): Imagen original en formato PIL.
        scales (list of float): Factores de escala (por ejemplo, [0.7071, 1.0, 1.4142]).

    Returns:
        list of PIL.Image: Lista de imágenes escaladas.
    """
    scaled_imgs = []
    w, h = img_pil.size
    for s in scales:
        new_size = (int(w * s), int(h * s))
        scaled_img = img_pil.resize(new_size, Image.BILINEAR)
        scaled_imgs.append(scaled_img)
    return scaled_imgs