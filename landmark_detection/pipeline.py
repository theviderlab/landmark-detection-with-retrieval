import torch
from landmark_detection.extract import CVNet_SG
from ultralytics import YOLO
import os
import cv2
from typing import List

import onnxruntime as ort
import onnx
from onnx import compose
from onnx import helper

class Pipeline_Yolo_CVNet_SG():

    def __init__(
        self,
        detector_file: str = "yolov8n-oiv7.onnx",
        extractor_onnx_file: str = "cvnet-sg.onnx",
        pipeline_onnx_file: str = "pipeline-yolo-cvnet-sg.onnx",
        image_dim: tuple[int] = (640, 640),
        allowed_classes: list[int] = [41,68,70,74,87,95,113,144,150,158,164,165,193,205,212,224,257,
                                      298,310,335,351,354,390,393,401,403,439,442,457,466,489,510,512,
                                      514,524,530,531,543,546,554,565,573,580,587,588,591],
        score_thresh: float = 0.10,
        iou_thresh: float = 0.45,
        scales: List[float] = [0.7071, 1.0, 1.4142],
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float]  = [0.229, 0.224, 0.225],
        rgem_pr: float   = 2.5,
        rgem_size: int   = 5,
        gem_p: float     = 4.6,
        sgem_ps: float   = 10.0,
        sgem_infinity: bool = False,
        eps: float       = 1e-8
    ):        
        self.image_dim = image_dim
        # Obtener el directorio donde está este archivo Python (el módulo)
        module_dir = os.path.dirname(os.path.abspath(__file__))

        # Si detector_file es .pt crea .onnx
        if not os.path.isabs(detector_file) and not os.path.exists(detector_file):
            # Construir la ruta absoluta dentro de este módulo
            detector_path = os.path.join(module_dir, "models", detector_file)
        else:
            detector_path = detector_file

        detector_file_ext = os.path.splitext(detector_path)[-1].lower()
        if detector_file_ext == ".pt":
            # Instanciar el detector de objetos
            print('Creando versión ONNX del detector')
            detector_pt = YOLO(detector_path)
            detector_onnx_path = self._export_detector(detector_pt)
        elif detector_file_ext == ".onnx":
            detector_onnx_path = detector_path
        else:
            raise ValueError(f"Extensión no soportada para detector_file: {detector_file_ext}")
        
        # Instanciar detector
        print('Instanciando el detector')
        self.detector = ort.InferenceSession(detector_onnx_path, providers=["CPUExecutionProvider"])

        # Si no existe el archivo .onnx se crea
        if not os.path.isabs(extractor_onnx_file) and not os.path.exists(extractor_onnx_file):
            # Construir la ruta absoluta dentro de este módulo
            extractor_onnx_path = os.path.join(module_dir, "models", extractor_onnx_file)
        else:
            extractor_onnx_path = extractor_onnx_file
        
        extractor = CVNet_SG(
            allowed_classes = allowed_classes,
            score_thresh = score_thresh,
            iou_thresh = iou_thresh,
            scales = scales,
            mean = mean,
            std = std,
            rgem_pr = rgem_pr,
            rgem_size = rgem_size,
            gem_p = gem_p,
            sgem_ps = sgem_ps,
            sgem_infinity = sgem_infinity,
            eps = eps
        ).eval()

        # Obtener el directorio donde está este archivo Python (el módulo)
        module_dir = os.path.dirname(os.path.abspath(__file__))
        # Construir la ruta absoluta a test_images/test.jpg dentro de este módulo
        test_image_path = os.path.join(module_dir, "test_images", "test.jpg")
        print('Creando versión ONNX del extractor')
        self._export_extractor(extractor, extractor_onnx_path, test_image_path)

        # Si no existe el archivo .onnx se crea
        if not os.path.isabs(pipeline_onnx_file) and not os.path.exists(pipeline_onnx_file):
            # Construir la ruta absoluta dentro de este módulo
            pipeline_onnx_path = os.path.join(module_dir, "models", pipeline_onnx_file)
        else:
            pipeline_onnx_path = pipeline_onnx_file
        
        print('Creando versión ONNX del pipeline completo')
        self._export_pipeline(detector_onnx_path, extractor_onnx_path, pipeline_onnx_path)

        # Instanciar extractor
        print('Instanciando el extractor')
        self.extractor = ort.InferenceSession(extractor_onnx_path, providers=["CPUExecutionProvider"])

        # Instanciar pipeline
        print('Instanciando el pipeline completo')
        self.pipeline = ort.InferenceSession(pipeline_onnx_path, providers=["CPUExecutionProvider"])

    def detect(self, image_path: str):

        # Cargar imagen
        img = self._load_image(image_path)

        # Obtener detecciones
        input_name = self.detector.get_inputs()[0].name
        detections = self.detector.run(None, {input_name: img})

        return detections
    
    def extract(self, img, detections):
        return self.extractor.run(None, {"detections": detections, "image": img})
    
    def run(self, image_path: str):
        # Cargar imagen
        img = self._load_image(image_path)

        # Obtener tamaño original de la imagen
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"No se encontró la imagen en {image_path}")
        orig_h, orig_w = img_bgr.shape[:2]

        # Ejecutar inferencia sobre el pipeline
        input_name = self.pipeline.get_inputs()[0].name
        results = self.pipeline.run(None, {input_name: img})

        # Convertir bounding boxes al tamaño original
        results = list(results)
        if len(results) > 0:
            results[0] = self._resize_boxes(results[0], (orig_w, orig_h))
        return results

    def _load_image(self, image_path: str):
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"No se encontró la imagen en {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Redimensionar a self.image_dim
        img_resized = cv2.resize(img_rgb, self.image_dim)

        # Convertir a tensor CHW y normalizar a [0,1] float32
        img_tensor = torch.from_numpy(img_resized).permute(2,0,1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # shape (1,3,self.image_dim[0],self.image_dim[1])

        return img_tensor.numpy()

    def _resize_boxes(self, boxes, original_size):
        """Ajusta las bounding boxes al tamaño original de la imagen."""
        orig_w, orig_h = original_size
        scale_x = orig_w / self.image_dim[0]
        scale_y = orig_h / self.image_dim[1]
        boxes = boxes.copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y
        return boxes

    def _export_detector(self, detector):
        # Exportar YOLO a ONNX
        paths = detector.export(
            format="onnx",
            imgsz=self.image_dim,
            opset=16,
            simplify=True
        )
        if isinstance(paths, (list, tuple)):
            detector_onnx_path = paths[-1]
        else:
            detector_onnx_path = paths

        # Cargar modelo en ONNX
        model = onnx.load(detector_onnx_path)
        graph = model.graph

        # Identificar el nombre del input de YOLO
        input_name = graph.input[0].name

        # Crear un nodo Identity que copie ese input a un nuevo tensor "images_out"
        identity_node = helper.make_node(
            "Identity",
            inputs=[input_name],      # la entrada original de YOLO (input_name)
            outputs=["images_out"],   # nuevo nombre de tensor que contendrá la misma imagen
            name="Identity_ExposeImages"
        )

        # Añadir ese nodo al final de graph.node
        graph.node.append(identity_node)

        # Declarar “images_out” como nuevo output del grafo, con la misma shape/dtipo que input_name
        # Podemos extraer la shape/dtipo del input para no recortarla a mano:
        input_type = graph.input[0].type.tensor_type.elem_type  # debería ser FLOAT (1)
        input_shape = []
        for dim in graph.input[0].type.tensor_type.shape.dim:
            # Si el dim_value > 0, lo tomamos; si es simbólico, dejamos dim_param
            if dim.dim_value > 0:
                input_shape.append(dim.dim_value)
            else:
                input_shape.append(dim.dim_param)

        new_output = helper.make_tensor_value_info(
            name="images_out",
            elem_type=input_type,
            shape=input_shape
        )
        graph.output.append(new_output)

        # Guardar el ONNX modificado en disco
        onnx.save(model, detector_onnx_path)

        return detector_onnx_path

    def _export_extractor(self, extractor, extractor_onnx_path: str, test_image_path: str):
        detections, img  = self.detect(test_image_path)

        img_tensor = torch.from_numpy(img)
        detections_tensor = torch.from_numpy(detections)

        # Exportar a ONNX
        torch.onnx.export(
            extractor,
            (detections_tensor, img_tensor),
            extractor_onnx_path,
            opset_version=16,                   # OPS versión >= 11 para NMS
            input_names=["detections", "image"],           # nombre del input
            output_names=["boxes", "scores", "classes", "descriptors"],
            dynamic_axes={
                "boxes":        {0: "num_boxes"},   
                "scores":       {0: "num_boxes"},
                "classes":      {0: "num_boxes"},
                "descriptors":  {0: "num_boxes"}
            },
            do_constant_folding=True
        )

    def _export_pipeline(self, detector_onnx_path: str, extractor_onnx_path: str, pipeline_onnx_path: str):
        detector_onnx = onnx.load(detector_onnx_path)
        extractor_onnx = onnx.load(extractor_onnx_path)

        merged_model = compose.merge_models(
            detector_onnx,
            extractor_onnx,
            io_map=[("images_out", "image"),("output0", "detections")]
        )

        onnx.checker.check_model(merged_model)  # valida topológica y esquemas
        onnx.save(merged_model, pipeline_onnx_path)