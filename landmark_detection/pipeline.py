import torch
from landmark_detection.extract import CVNet_SG
from landmark_detection.preprocess import PreprocessModule
from landmark_detection.postprocess import PostprocessModule

from ultralytics import YOLO
import os
import cv2
from typing import List
import json

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
        self.preprocess_module = PreprocessModule(image_dim)
        self.postprocess_module = PostprocessModule(image_dim)
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

        # Almacenar parámetros de configuración
        self.detector_file = detector_onnx_path
        self.extractor_onnx_file = extractor_onnx_path
        self.pipeline_onnx_file = pipeline_onnx_path
        self.allowed_classes = allowed_classes
        self.score_thresh = score_thresh
        self.iou_thresh = iou_thresh
        self.scales = scales
        self.mean = mean
        self.std = std
        self.rgem_pr = rgem_pr
        self.rgem_size = rgem_size
        self.gem_p = gem_p
        self.sgem_ps = sgem_ps
        self.sgem_infinity = sgem_infinity
        self.eps = eps

    def detect(self, image):

        # Preprocesar imagen
        img, orig_size = self.preprocess(image)

        # Obtener detecciones
        detector_inputs = {self.detector.get_inputs()[0].name: img}
        if len(self.detector.get_inputs()) > 1:
            detector_inputs[self.detector.get_inputs()[1].name] = orig_size
        detections = self.detector.run(None, detector_inputs)

        return detections, img, orig_size
    
    def extract(self, img, detections, orig_size=None):
        extractor_inputs = {"detections": detections, "image": img}
        if orig_size is not None and len(self.extractor.get_inputs()) > 2:
            extractor_inputs[self.extractor.get_inputs()[2].name] = orig_size
        return self.extractor.run(None, extractor_inputs)
    
    def run(self, image):
        # Preprocesar la imagen
        img, orig_size = self.preprocess(image)

        # Ejecutar inferencia sobre el pipeline
        pipeline_inputs = {self.pipeline.get_inputs()[0].name: img}
        if len(self.pipeline.get_inputs()) > 1:
            pipeline_inputs[self.pipeline.get_inputs()[1].name] = orig_size
        results = self.pipeline.run(None, pipeline_inputs)

        # Postprocesar resultados
        return self.postprocess(results, orig_size)

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
            inputs=[input_name],
            outputs=["images_out"],
            name="Identity_ExposeImages",
        )

        # Nodo para propagar orig_size sin modificar
        orig_node = helper.make_node(
            "Identity",
            inputs=["orig_size"],
            outputs=["orig_size_out"],
            name="Identity_ExposeOrigSize",
        )

        # Añadir los nodos al grafo
        graph.node.extend([identity_node, orig_node])

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

        # Declarar el nuevo input y output para orig_size
        orig_input = helper.make_tensor_value_info(
            name="orig_size",
            elem_type=onnx.TensorProto.FLOAT,
            shape=[2],
        )
        graph.input.append(orig_input)

        orig_output = helper.make_tensor_value_info(
            name="orig_size_out",
            elem_type=onnx.TensorProto.FLOAT,
            shape=[2],
        )
        graph.output.append(orig_output)

        # Guardar el ONNX modificado en disco
        onnx.save(model, detector_onnx_path)

        return detector_onnx_path

    def _export_extractor(self, extractor, extractor_onnx_path: str, test_image_path: str):
        detections, img, _ = self.detect(test_image_path)

        img_tensor = torch.from_numpy(img)
        if isinstance(detections, (list, tuple)):
            detections = detections[0]
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

        # Añadir bypass para orig_size
        model = onnx.load(extractor_onnx_path)
        graph = model.graph

        orig_input = helper.make_tensor_value_info(
            name="orig_size",
            elem_type=onnx.TensorProto.FLOAT,
            shape=[2],
        )
        graph.input.append(orig_input)

        orig_node = helper.make_node(
            "Identity",
            inputs=["orig_size"],
            outputs=["orig_size_out"],
            name="Identity_ExposeOrigSize",
        )
        graph.node.append(orig_node)

        orig_output = helper.make_tensor_value_info(
            name="orig_size_out",
            elem_type=onnx.TensorProto.FLOAT,
            shape=[2],
        )
        graph.output.append(orig_output)

        onnx.save(model, extractor_onnx_path)

    def _export_pipeline(self, detector_onnx_path: str, extractor_onnx_path: str, pipeline_onnx_path: str):
        detector_onnx = onnx.load(detector_onnx_path)
        extractor_onnx = onnx.load(extractor_onnx_path)

        merged_model = compose.merge_models(
            detector_onnx,
            extractor_onnx,
            io_map=[
                ("images_out", "image"),
                ("output0", "detections"),
                ("orig_size_out", "orig_size"),
            ]
        )

        onnx.checker.check_model(merged_model)  # valida topológica y esquemas
        onnx.save(merged_model, pipeline_onnx_path)

    def to_json(self, json_path: str):
        """Guarda la configuración del pipeline en un archivo JSON."""
        config = {
            "detector_file": self.detector_file,
            "extractor_onnx_file": self.extractor_onnx_file,
            "pipeline_onnx_file": self.pipeline_onnx_file,
            "image_dim": list(self.image_dim),
            "allowed_classes": self.allowed_classes,
            "score_thresh": self.score_thresh,
            "iou_thresh": self.iou_thresh,
            "scales": self.scales,
            "mean": self.mean,
            "std": self.std,
            "rgem_pr": self.rgem_pr,
            "rgem_size": self.rgem_size,
            "gem_p": self.gem_p,
            "sgem_ps": self.sgem_ps,
            "sgem_infinity": self.sgem_infinity,
            "eps": self.eps,
        }

        dir_name = os.path.dirname(json_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
            except Exception:
                data = {}
            if isinstance(data, list):
                data.append(config)
            elif isinstance(data, dict):
                data.update(config)
            else:
                data = config
        else:
            data = config

        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

    def preprocess(self, image):
        """Carga y normaliza la imagen."""
        if isinstance(image, str):
            img_bgr = cv2.imread(image)
            if img_bgr is None:
                raise FileNotFoundError(f"No se encontró la imagen en {image}")
            img_tensor = torch.from_numpy(img_bgr)
        else:
            img_tensor = torch.as_tensor(image)

        processed, orig_size = self.preprocess_module(img_tensor)
        return processed.numpy(), (int(orig_size[0]), int(orig_size[1]))

    def postprocess(self, results, original_size):
        """Escala las cajas al tamaño original de la imagen."""
        results = list(results)
        if len(results) > 0:
            boxes = torch.as_tensor(results[0])
            scores = torch.as_tensor(results[1])
            classes = torch.as_tensor(results[2])
            descriptors = torch.as_tensor(results[3])
            orig = torch.tensor([
                original_size[0],
                original_size[1],
            ], dtype=torch.float32)
            (
                results[0],
                results[1],
                results[2],
                results[3],
            ) = self.postprocess_module(boxes, scores, classes, descriptors, orig)
            results = [r.numpy() for r in results]
        return results
