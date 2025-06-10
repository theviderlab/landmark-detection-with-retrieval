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
        
        preprocess_onnx_path = os.path.join(module_dir, "models", "preprocess.onnx")
        postprocess_onnx_path = os.path.join(module_dir, "models", "postprocess.onnx")

        print('Creando versión ONNX del preprocess')
        self._export_preprocess(self.preprocess_module, preprocess_onnx_path, test_image_path)

        print('Creando versión ONNX del postprocess')
        self._export_postprocess(self.postprocess_module, postprocess_onnx_path)

        print('Creando versión ONNX del pipeline completo')
        self._export_pipeline(
            preprocess_onnx_path,
            detector_onnx_path,
            extractor_onnx_path,
            postprocess_onnx_path,
            pipeline_onnx_path,
        )

        if os.path.exists(preprocess_onnx_path):
            os.remove(preprocess_onnx_path)
        if os.path.exists(postprocess_onnx_path):
            os.remove(postprocess_onnx_path)

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
        detections = self.detector.run(None, detector_inputs)

        return detections, img, orig_size
    
    def extract(self, img, detections):
        extractor_inputs = {"detections": detections, "image": img}
        return self.extractor.run(None, extractor_inputs)
    
    def run(self, image):
        """Ejecuta la inferencia completa empleando el modelo ONNX unido."""
        if isinstance(image, str):
            img_bgr = cv2.imread(image)
            if img_bgr is None:
                raise FileNotFoundError(f"No se encontró la imagen en {image}")
            img_tensor = torch.from_numpy(img_bgr)
        else:
            img_tensor = torch.as_tensor(image)

        orig = torch.tensor([
            img_tensor.shape[1],
            img_tensor.shape[0],
        ], dtype=torch.float32)
        pipeline_inputs = {
            self.pipeline.get_inputs()[0].name: img_tensor.numpy(),
            self.pipeline.get_inputs()[1].name: orig.numpy(),
        }
        results = self.pipeline.run(None, pipeline_inputs)
        return results

    def _export_detector(self, detector):
        # Exportar YOLO a ONNX
        paths = detector.export(
            format="onnx",
            imgsz=self.image_dim,
            opset=16,
            simplify=True,
        )
        if isinstance(paths, (list, tuple)):
            detector_onnx_path = paths[-1]
        else:
            detector_onnx_path = paths

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


    def _export_preprocess(self, preprocess_module, preprocess_onnx_path: str, test_image_path: str):
        img_bgr = cv2.imread(test_image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"No se encontró {test_image_path}")
        img_tensor = torch.from_numpy(img_bgr)

        dummy_orig = torch.tensor([float(img_tensor.shape[1]), float(img_tensor.shape[0])], dtype=torch.float32)

        torch.onnx.export(
            preprocess_module,
            (img_tensor, dummy_orig),
            preprocess_onnx_path,
            opset_version=16,
            input_names=["image_bgr", "orig_size"],
            output_names=["image", "orig_size"],
            dynamic_axes={"image_bgr": {0: "h", 1: "w"}},
            do_constant_folding=True,
        )

    def _export_postprocess(self, postprocess_module, postprocess_onnx_path: str):
        dummy_boxes = torch.zeros((1, 4), dtype=torch.float32)
        dummy_scores = torch.zeros((1,), dtype=torch.float32)
        dummy_classes = torch.zeros((1,), dtype=torch.int64)
        dummy_descriptors = torch.zeros((1, 2048), dtype=torch.float32)
        dummy_orig = torch.tensor([float(self.image_dim[0]), float(self.image_dim[1])], dtype=torch.float32)

        torch.onnx.export(
            postprocess_module,
            (
                dummy_boxes,
                dummy_scores,
                dummy_classes,
                dummy_descriptors,
                dummy_orig,
            ),
            postprocess_onnx_path,
            opset_version=16,
            input_names=[
                "final_boxes",
                "final_scores",
                "final_classes",
                "descriptors",
                "orig_size",
            ],
            output_names=["boxes", "scores", "classes", "descriptors"],
            dynamic_axes={
                "final_boxes": {0: "num_boxes"},
                "final_scores": {0: "num_boxes"},
                "final_classes": {0: "num_boxes"},
                "descriptors": {0: "num_boxes"},
            },
            do_constant_folding=True,
        )

    def _export_pipeline(
        self,
        preprocess_onnx_path: str,
        detector_onnx_path: str,
        extractor_onnx_path: str,
        postprocess_onnx_path: str,
        pipeline_onnx_path: str,
    ):
        preprocess_onnx = onnx.load(preprocess_onnx_path)
        detector_onnx = onnx.load(detector_onnx_path)
        extractor_onnx = onnx.load(extractor_onnx_path)
        postprocess_onnx = onnx.load(postprocess_onnx_path)

        # Prefix graphs to avoid name collisions when merging
        detector_onnx = compose.add_prefix(detector_onnx, "det_")
        extractor_onnx = compose.add_prefix(extractor_onnx, "ext_")
        postprocess_onnx = compose.add_prefix(postprocess_onnx, "post_")

        # Preprocess -> Detector
        det_input_name = detector_onnx.graph.input[0].name
        det_output_names = [o.name for o in detector_onnx.graph.output]
        merged_pd = compose.merge_models(
            preprocess_onnx,
            detector_onnx,
            io_map=[
                ("image", det_input_name),
            ],
            outputs=det_output_names + ["image"],
        )

        # Detector -> Extractor
        ext_inputs = [i.name for i in extractor_onnx.graph.input]
        det_outputs = [o.name for o in detector_onnx.graph.output]
        merged_pde = compose.merge_models(
            merged_pd,
            extractor_onnx,
            io_map=[
                (det_outputs[0], ext_inputs[0]),  # detections
                ("image", ext_inputs[1]),         # preprocessed image
            ],
        )

        # Extractor -> Postprocess
        ext_outputs = [o.name for o in extractor_onnx.graph.output]
        post_inputs = [i.name for i in postprocess_onnx.graph.input]
        merged_model = compose.merge_models(
            merged_pde,
            postprocess_onnx,
            io_map=[
                (ext_outputs[0], post_inputs[0]),
                (ext_outputs[1], post_inputs[1]),
                (ext_outputs[2], post_inputs[2]),
                (ext_outputs[3], post_inputs[3]),
            ],
        )

        # Añadir entrada orig_size directamente al modelo combinado
        orig_input = helper.make_tensor_value_info(
            name="orig_size",
            elem_type=onnx.TensorProto.FLOAT,
            shape=[2],
        )
        merged_model.graph.input.append(orig_input)

        # Conectar orig_size a la etapa de postprocesado y eliminar el input
        # prefijado generado al fusionar los modelos
        for node in merged_model.graph.node:
            for i, inp in enumerate(node.input):
                if inp == post_inputs[4]:
                    node.input[i] = "orig_size"

        for i, inp in enumerate(list(merged_model.graph.input)):
            if inp.name == post_inputs[4]:
                del merged_model.graph.input[i]
                break

        onnx.checker.check_model(merged_model)
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
        orig_size = orig_size.to(dtype=torch.float32).numpy()
        return processed.numpy(), orig_size

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
            results = [r for r in results]
        return results
