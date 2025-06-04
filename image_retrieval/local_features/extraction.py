import numpy as np
import tensorflow as tf
from skimage.measure import label, regionprops
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

"""
Módulo para extracción de características region-based mediante pooling basado en regiones.

Esta clase se inicializa con un modelo Keras ya cargado y una configuración que incluye parámetros
como el número de regiones a seleccionar, las capas para extraer las activaciones (feat_layer y mask_layer),
y que permite visualizar las regiones (bounding boxes) sobre la imagen original con un heatmap que indica la importancia.

En base al paper:

Chen, Z., Maffra, F., Sa, I., & Chli, M. (2017). 
Only look once, mining distinctive landmarks from ConvNet for visual place recognition. 
2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 135, 9–16. 
https://doi.org/10.1109/IROS.2017.8202131
"""

class RegionBasedExtractor:
    def __init__(self, model, config=None):
        """
        Inicializa el extractor con un modelo ya cargado y una configuración opcional.

        Args:
            model: Instancia de un modelo Keras ya cargado.
            config (dict, opcional): Diccionario de configuración con parámetros. Ejemplo:
                {
                    "M": 200,
                    "feat_layer": "conv4_block6_out",
                    "mask_layer": "conv4_block6_out"
                }
        """
        self.model = model
        if config is None:
            config = {}
        self.M = config.get("M", 200)
        self.feat_layer = config.get("feat_layer", "conv4_block6_out")
        self.mask_layer = config.get("mask_layer", "conv4_block6_out")
        # Atributos para almacenar los resultados de la extracción para visualización
        self.final_box = None       # Bounding boxes (en el mapa de activación)
        self.final_mask = None      # Índices de canal correspondientes a cada región
        self.final_importance = None  # Importancia (mean intensity) de cada región
        self.map_shape = None       # Forma (H, W) del mapa de activación

    def _encode_feat_region_based(self, feat, mask):
        """
        Realiza el pooling basado en regiones a partir de los mapas de activación de una capa de características (feat)
        y una máscara (mask). Se asume que ambos tienen forma (H, W, C), donde C es el número de canales.

        Retorna:
            output: numpy array de tamaño (M, d_local), donde M es el número de regiones seleccionadas 
                    y d_local es la dimensión del vector de características (por ejemplo, el número de canales de feat).
            final_box: numpy array de tamaño (M, 4) con las bounding boxes en formato [x, y, w, h]
                    (coordenadas en el mapa de activación).
            final_mask: numpy array de tamaño (M,) con los índices de canal correspondientes a cada región.
            final_importance: numpy array de tamaño (M,) con la importancia (por ejemplo, mean intensity) de cada región.
        """
        # Convertir a numpy si las entradas son tensores
        if isinstance(feat, tf.Tensor):
            feat = feat.numpy()
        if isinstance(mask, tf.Tensor):
            mask = mask.numpy()

        H, W, C = mask.shape
        M = self.M

        allmean = []
        allbox = []
        whichmask = []

        # Se itera sobre cada canal del mapa de máscara
        for mask_idx in range(C):
            temp_mask = mask[:, :, mask_idx]
            thresh = temp_mask.min()  # Umbral: el mínimo valor de la máscara
            binary_mask = temp_mask > thresh  # Crear máscara binaria

            # Etiquetar componentes conectados (conectividad 8)
            labeled_mask = label(binary_mask, connectivity=2)
            props = regionprops(labeled_mask, intensity_image=temp_mask)
            for prop in props:
                allmean.append(prop.mean_intensity)
                # La bbox viene en formato (min_row, min_col, max_row, max_col)
                min_row, min_col, max_row, max_col = prop.bbox  
                width = max_col - min_col
                height = max_row - min_row
                allbox.append([min_col, min_row, width, height])  # Guardar en formato [x, y, w, h]
                whichmask.append(mask_idx)

        allmean = np.array(allmean)
        allbox = np.array(allbox)
        whichmask = np.array(whichmask)

        # Seleccionar las regiones con mayor intensidad (pooling de las top M regiones)
        sorted_mean = np.sort(allmean)[::-1]
        threshold_val = sorted_mean[M - 1] if len(sorted_mean) >= M else sorted_mean[-1]
        upper = np.where(allmean >= threshold_val)[0]
        final_box = allbox[upper, :]      # Bounding boxes seleccionadas
        final_mask = whichmask[upper]      # Índices de canal correspondientes
        final_importance = allmean[upper]  # Importancia de cada región

        # Realizar el pooling ponderado sobre cada región
        output = []
        for i in range(len(final_mask)):
            region_box = final_box[i]
            x, y, w, h = region_box.astype(int)
            channel_idx = final_mask[i]
            # Extraer la región de características (para todos los canales)
            pool_feat_region = feat[y:y+h, x:x+w, :]
            # Extraer la región de la máscara correspondiente al canal específico
            pool_mask = mask[y:y+h, x:x+w, channel_idx]

            # Reorganizar a forma (d_local, h, w)
            pool_feat_region = np.transpose(pool_feat_region, (2, 0, 1))
            d_local = pool_feat_region.shape[0]
            pool_feat_flat = pool_feat_region.reshape(d_local, -1)

            # Aplanar la máscara y normalizarla
            flatten_pool = pool_mask.flatten()
            norm_val = np.linalg.norm(flatten_pool)
            if norm_val != 0:
                flatten_pool = flatten_pool / norm_val

            # Producto punto ponderado
            pool_multi = pool_feat_flat.dot(flatten_pool)
            norm_pool_multi = np.linalg.norm(pool_multi)
            if norm_pool_multi != 0:
                pool_multi = pool_multi / norm_pool_multi

            output.append(pool_multi.reshape(-1, 1))

        if output:
            # Concatenar las columnas: originalmente se obtiene un array de forma (d_local, M).
            # Para obtener (M, d_local), se transpone el resultado.
            output = np.concatenate(output, axis=1).T
        else:
            output = np.array([])

        return output, final_box, final_mask, final_importance

    def extract(self, preprocessed_img):
        """
        Extrae características region-based de una sola imagen preprocesada.

        Args:
            preprocessed_img (np.ndarray): Tensor con forma (1, H, W, C)

        Returns:
            np.ndarray: Array con forma (M, d_local) con características locales extraídas.
        """
        # Construir modelo dual para obtener las activaciones de las dos capas especificadas
        dual_output_model = tf.keras.models.Model(
            self.model.inputs,
            [self.model.get_layer(self.feat_layer).output, self.model.get_layer(self.mask_layer).output]
        )
        feat_out, mask_out = dual_output_model(preprocessed_img, training=False)
        # Extraer la salida quitando la dimensión de batch; se asume formato (H, W, C)
        feat_map = feat_out[0]
        mask_map = mask_out[0]
        # Almacenar la forma del mapa de activación para usar en visualización
        self.map_shape = mask_map.shape[:2]
        features, final_box, final_mask, final_importance = self._encode_feat_region_based(feat_map, mask_map)
        self.final_box = final_box
        self.final_mask = final_mask
        self.final_importance = final_importance
        return features

    def visualize(self, vis_img, alpha=0.05):
        """
        Visualiza las bounding boxes (almacenadas tras llamar a extract) sobre la imagen original.
        Además, pinta el interior de cada región como un heatmap basado en la importancia y añade una barra de color.
        Las regiones se ordenan de menor a mayor importancia para que las más importantes se dibujen por encima.

        Args:
            vis_img: Imagen original (PIL).
        """
        if self.final_box is None or self.map_shape is None:
            print("No hay regiones para visualizar. Asegúrate de llamar a extract() primero.")
            return

        H_map, W_map = self.map_shape
        orig_width, orig_height = vis_img.size
        scale_x = orig_width / W_map
        scale_y = orig_height / H_map

        # Convertir la imagen a formato BGR para OpenCV
        vis_img_cv = cv2.cvtColor(np.array(vis_img), cv2.COLOR_RGB2BGR)

        # Normalizar las importancias para asignar colores en el heatmap
        if self.final_importance is not None and len(self.final_importance) > 0:
            min_imp = np.min(self.final_importance)
            max_imp = np.max(self.final_importance)
        else:
            min_imp, max_imp = 0, 1

        colormap = cm.get_cmap('jet')

        # Obtener el orden de índices (de menor a mayor importancia)
        order = np.argsort(self.final_importance)

        # Iterar sobre las regiones en el orden especificado
        for idx in order:
            bbox = self.final_box[idx]
            imp = self.final_importance[idx]
            x, y, w, h = bbox.astype(int)
            x_scaled = int(x * scale_x)
            y_scaled = int(y * scale_y)
            w_scaled = int(w * scale_x)
            h_scaled = int(h * scale_y)

            # Normalizar la importancia a un valor entre 0 y 1
            norm_val = (imp - min_imp) / (max_imp - min_imp) if (max_imp - min_imp) != 0 else 0.5

            # Obtener el color correspondiente del colormap (convertido a formato BGR)
            color_rgba = colormap(norm_val)
            color = (int(color_rgba[2]*255), int(color_rgba[1]*255), int(color_rgba[0]*255))

            # Crear overlay para pintar el interior de la región
            overlay = vis_img_cv.copy()
            cv2.rectangle(overlay, (x_scaled, y_scaled), (x_scaled + w_scaled, y_scaled + h_scaled), color, -1)
            vis_img_cv = cv2.addWeighted(overlay, alpha, vis_img_cv, 1 - alpha, 0)

            # Dibujar el borde en blanco
            cv2.rectangle(vis_img_cv, (x_scaled, y_scaled), (x_scaled + w_scaled, y_scaled + h_scaled), (255, 255, 255), 2)

        # Convertir la imagen resultante a RGB para mostrar con matplotlib
        vis_img_rgb = cv2.cvtColor(vis_img_cv, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(vis_img_rgb)
        ax.set_title("Regiones seleccionadas con heatmap interior")
        ax.axis("off")

        # Crear barra de color para la leyenda
        norm = mcolors.Normalize(vmin=min_imp, vmax=max_imp)
        sm = cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="Importancia (mean intensity)")

        plt.show()
