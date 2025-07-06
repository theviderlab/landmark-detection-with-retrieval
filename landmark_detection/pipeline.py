import torch
from landmark_detection.preprocess import PreprocessModule
from landmark_detection.extract import CVNet_SG
from landmark_detection.postprocess import PostprocessModule
from landmark_detection.search import Similarity_Search

from ultralytics import YOLO
import os
import cv2
from typing import List, Tuple
import json
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm

import onnxruntime as ort
import onnx
from onnx import compose
from onnx import helper

class Pipeline_Landmark_Detection():

    def __init__(
        self,
        detector_file: str = "detector.onnx",
        extractor_onnx_file: str = "extractor.onnx",
        searcher_onnx_file: str = "searcher.onnx",
        pipeline_onnx_file: str = "pipeline.onnx",
        image_dim: tuple[int] = (640, 640),                                                             # detection-preprocess
        allowed_classes: list[int] = [41,68,70,74,87,95,113,144,150,158,164,165,193,205,212,224,257,
                                      298,310,335,351,390,393,401,403,439,442,457,466,489,510,512,
                                      514,524,530,531,543,546,554,565,573,580,587,588,591],             # detection-postprocess
        score_thresh: float = 0.10,                                                                     # detection-postprocess
        iou_thresh: float = 0.45,                                                                       # detection-postprocess
        scales: list[float] = [0.7071, 1.0, 1.4142],                                                    # extraction-preprocess
        mean: list[float] = [0.485, 0.456, 0.406],                                                      # extraction-preprocess
        std: list[float]  = [0.229, 0.224, 0.225],                                                      # extraction-preprocess
        rgem_pr: float   = 2.5,                                                                         # pooling
        rgem_size: int   = 5,                                                                           # pooling
        gem_p: float     = 4.6,                                                                         # pooling
        sgem_ps: float   = 10.0,                                                                        # pooling
        sgem_infinity: bool = False,                                                                    # pooling
        eps: float       = 1e-8,                                                                        # pooling
        topk: int = 5,                                                                                  # search-postprocess
        min_sim: float = 0.8,                                                                           # search-postprocess
        min_votes: float = 0.0,                                                                         # search-postprocess
        remove_inner_boxes: float | None = None,                                                        # search-postprocess
        join_boxes: bool = False,                                                                       # search-postprocess
    ):
        # Almacenar parámetros de configuración
        self.image_dim = image_dim
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
        self.topk = topk
        self.min_sim = min_sim
        self.min_votes = min_votes
        self.remove_inner_boxes = remove_inner_boxes
        self.join_boxes = join_boxes

        # Cargar los datos de prueba
        module_dir = os.path.dirname(os.path.abspath(__file__))
        test_image_path = os.path.join(module_dir, "test_data", "test.jpg")
        test_places_db_path = os.path.join(module_dir, "test_data", "test_places_db.pkl")

        with open(test_places_db_path, 'rb') as f:
            places_db = pickle.load(f)
        places_db = places_db if isinstance(places_db, torch.Tensor) else torch.tensor(places_db, dtype=torch.float32)

        # Preprocess

        print('Creando versión ONNX del preprocess')
        self.preprocess_onnx_path = os.path.join(module_dir, "models", "preprocess.onnx")
        self.preprocess_module = PreprocessModule(self.image_dim)
        self._export_preprocess(self.preprocess_module, self.preprocess_onnx_path, test_image_path)

        print('Instanciando el preprocessor')
        self.preprocessor = ort.InferenceSession(self.preprocess_onnx_path, providers=["CPUExecutionProvider"])

        # Detector

        if not os.path.isabs(detector_file) and not os.path.exists(detector_file):
            # Construir la ruta absoluta dentro de este módulo
            detector_path = os.path.join(module_dir, "models", detector_file)
        else:
            detector_path = detector_file

        detector_file_ext = os.path.splitext(detector_path)[-1].lower()
        if detector_file_ext == ".pt": # Si detector_file es .pt crea .onnx
            print('Creando versión ONNX del detector')
            detector_pt = YOLO(detector_path)
            self.detector_onnx_path = self._export_detector(detector_pt)
        elif detector_file_ext == ".onnx":
            self.detector_onnx_path = detector_path
        else:
            raise ValueError(f"Extensión no soportada para detector_file: {detector_file_ext}")

        print('Instanciando el detector')
        self.detector = ort.InferenceSession(self.detector_onnx_path, providers=["CPUExecutionProvider"])

        # Extractor

        # Si no existe el archivo .onnx se crea
        if not os.path.isabs(extractor_onnx_file) and not os.path.exists(extractor_onnx_file):
            # Construir la ruta absoluta dentro de este módulo
            self.extractor_onnx_path = os.path.join(module_dir, "models", extractor_onnx_file)
        else:
            self.extractor_onnx_path = extractor_onnx_file
        
        extractor = CVNet_SG(
            allowed_classes = self.allowed_classes,
            score_thresh = self.score_thresh,
            iou_thresh = self.iou_thresh,
            scales = self.scales,
            mean = self.mean,
            std = self.std,
            rgem_pr = self.rgem_pr,
            rgem_size = self.rgem_size,
            gem_p = self.gem_p,
            sgem_ps = self.sgem_ps,
            sgem_infinity = self.sgem_infinity,
            eps = self.eps
        ).eval()

        print('Creando versión ONNX del extractor')
        self._export_extractor(extractor, self.extractor_onnx_path, test_image_path, places_db)

        print('Instanciando el extractor')
        self.extractor = ort.InferenceSession(self.extractor_onnx_path, providers=["CPUExecutionProvider"])

        # Searcher

        # Si no existe el archivo .onnx se crea
        if not os.path.isabs(searcher_onnx_file) and not os.path.exists(searcher_onnx_file):
            # Construir la ruta absoluta dentro de este módulo
            self.searcher_onnx_path = os.path.join(module_dir, "models", searcher_onnx_file)
        else:
            self.searcher_onnx_path = searcher_onnx_file

        searcher = Similarity_Search(
            topk = self.topk,
            min_sim = self.min_sim,
            min_votes = self.min_votes,
            remove_inner_boxes = self.remove_inner_boxes,
            join_boxes = self.join_boxes,
        )

        print('Creando versión ONNX del searcher')
        self._export_searcher(searcher, self.searcher_onnx_path, test_image_path, places_db)

        print('Instanciando el searcher')
        self.searcher = ort.InferenceSession(self.searcher_onnx_path, providers=["CPUExecutionProvider"])

        # Postprocess

        print('Creando versión ONNX del postprocess')
        self.postprocess_onnx_path = os.path.join(module_dir, "models", "postprocess.onnx")
        self.postprocess_module = PostprocessModule(self.image_dim)
        self._export_postprocess(self.postprocess_module, self.postprocess_onnx_path, test_image_path, places_db)

        print('Instanciando el postprocessor')
        self.postprocessor = ort.InferenceSession(self.postprocess_onnx_path, providers=["CPUExecutionProvider"])

        # Pipeline

        print('Creando versión ONNX del pipeline completo')
        # Si no existe el archivo .onnx se crea
        if not os.path.isabs(pipeline_onnx_file) and not os.path.exists(pipeline_onnx_file):
            # Construir la ruta absoluta dentro de este módulo
            self.pipeline_onnx_path = os.path.join(module_dir, "models", pipeline_onnx_file)
        else:
            self.pipeline_onnx_path = pipeline_onnx_file

        self._export_pipeline(
            self.preprocess_onnx_path,
            self.detector_onnx_path,
            self.extractor_onnx_path,
            self.searcher_onnx_path,
            self.postprocess_onnx_path,
            self.pipeline_onnx_path,
        )

        # Instanciar pipeline
        print('Instanciando el pipeline completo')
        self.pipeline = ort.InferenceSession(self.pipeline_onnx_path, providers=["CPUExecutionProvider"])

    def preprocess(self, image, places_db):
        """
        Carga y normaliza la imagen.
                
        Args:
            image (numpy.array | str): Imagen como Numpy array o un string con el path a la imagen.
            places_db (list[float64]): Descriptores de la base de datos.

        Returns:
            image_proc (numpy.array[float32]): Imagen preprocesada.
            orig_size (numpy.array[int]): Dimensiones de la imagen original.
            places_db (list[float64]): Descriptores de la base de datos.
        """
        if isinstance(image, str):
            image_bgr = cv2.imread(image)
            if image_bgr is None:
                raise FileNotFoundError(f"No se encontró la imagen en {image}")
        else:
            image_bgr = np.asarray(image, dtype=np.float32)

        if isinstance(places_db, torch.Tensor):
            places_np = places_db.detach().cpu().numpy().astype(np.float32)
        else:
            places_np = np.asarray(places_db, dtype=np.float32)

        preprocessor_inputs = {"image_bgr": image_bgr, "places_db": places_np}
        return self.preprocessor.run(None, preprocessor_inputs)
    
    def detect(self, image_proc, places_db, orig_size):
        """
        Realiza la detección de objetos sobre la imagen.
                
        Args:
            image_proc (numpy.array): Imagen preprocesada.
            places_db (list[float64]): Descriptores de la base de datos.
            orig_size (numpy.array[int]): Dimensiones de la imagen original.

        Returns:
            detections (numpy.array[float32]): Detecciones [1, 4 + C, N].
            image_proc (numpy.array[float32]): Imagen preprocesada (bypass).
            places_db (list[float64]): Descriptores de la base de datos (bypass).
            orig_size (list[int]): Dimensiones de la imagen original (bypass).
        """
        if isinstance(places_db, torch.Tensor):
            places_np = places_db.detach().cpu().numpy().astype(np.float32)
        else:
            places_np = np.asarray(places_db, dtype=np.float32)

        detector_inputs = {"images": image_proc, "places_db": places_np, "orig_size": orig_size}
        return self.detector.run(None, detector_inputs)
    
    def extract(self, detections, image_proc, places_db, orig_size):
        """
        Crea los embeddings sobre las detecciones.
                
        Args:
            detections (list[float]): Detecciones [1, 4 + C, N].
            image_proc (numpy.array): Imagen preprocesada.
            places_db (list[float64]): Descriptores de la base de datos.
            orig_size (numpy.array[int]): Dimensiones de la imagen original.

        Returns:
            boxes (numpy.array[float32]): Cajas de los objetos detectados.
            descriptors (numpy.array[float32]): Descriptores de los objetos detectados.
            places_db (list[float64]): Descriptores de la base de datos (bypass).
            orig_size (list[int]): Dimensiones de la imagen original (bypass).
        """
        if isinstance(places_db, torch.Tensor):
            places_np = places_db.detach().cpu().numpy().astype(np.float32)
        else:
            places_np = np.asarray(places_db, dtype=np.float32)

        extractor_inputs = {"detections": detections, "image": image_proc, "places_db": places_np, "orig_size": orig_size}
        return self.extractor.run(None, extractor_inputs)

    def search(self, boxes, descriptors, places_db, orig_size):
        """
        Realiza la búsqueda de los objetos detectados en la base de datos.
                
        Args:
            boxes (numpy.array): Cajas de los objetos detectados.
            descriptors (numpy.array): Descriptores de los objetos detectados.
            places_db (list[float64]): Descriptores de la base de datos (bypass).
            orig_size (list[int]): Dimensiones de la imagen original (bypass).

        Returns: 
            boxes_out (numpy.array[float32]): Cajas con lugares identificados.
            scores_out (numpy.array[float32]): Similitudes de los lugares identificados.
            classes_out (list[int64]): Lugares identificados.
            orig_size (list[int]): Dimensiones de la imagen original (bypass).
        """
        if isinstance(places_db, torch.Tensor):
            places_np = places_db.detach().cpu().numpy().astype(np.float32)
        else:
            places_np = np.asarray(places_db, dtype=np.float32)

        searcher_inputs = {"boxes": boxes, "descriptors": descriptors, "places_db": places_np, "orig_size": orig_size}
        return self.searcher.run(None, searcher_inputs)

    def postprocess(self, boxes, scores, classes, orig_size):
        """
        Realiza el reescalado de las cajas al tamaño de la imagen original.
                
        Args:
            boxes (numpy.array[float32]): Cajas con lugares identificados.
            scores (numpy.array[float32]): Similitudes de los lugares identificados.
            classes (list[int64]): Lugares identificados.
            orig_size (list[int]): Dimensiones de la imagen original (bypass).

        Returns: 
            final_boxes (numpy.array[float32]): Cajas con lugares identificados.
            final_scores (numpy.array[float32]): Similitudes de los lugares identificados.
            final_classes (list[int64]): Lugares identificados.
        """

        postprocessor_inputs = {"final_boxes": boxes, "final_scores": scores, "final_classes": classes, "orig_size": orig_size}
        return self.postprocessor.run(None, postprocessor_inputs)
    
    def run(self, image, places_db):
        """Ejecuta la inferencia completa empleando el modelo ONNX unido."""
        if isinstance(image, str):
            img_bgr = cv2.imread(image)
            if img_bgr is None:
                raise FileNotFoundError(f"No se encontró la imagen en {image}")
        else:
            img_bgr = image

        img_tensor = torch.as_tensor(img_bgr)

        if isinstance(places_db, torch.Tensor):
            db_np = places_db.detach().cpu().numpy().astype(np.float32)
        else:
            db_np = np.asarray(places_db, dtype=np.float32)

        pipeline_inputs = {
            self.pipeline.get_inputs()[0].name: img_tensor.numpy(),
            self.pipeline.get_inputs()[1].name: db_np,
        }
        results = self.pipeline.run(None, pipeline_inputs)

        return results
    
    def _export_preprocess(self, preprocessor, preprocess_onnx_path: str, test_image_path: str):
        img_bgr = cv2.imread(test_image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"No se encontró {test_image_path}")
        img_tensor = torch.from_numpy(img_bgr)

        export_args = dict(
            opset_version=16,
            input_names=["image_bgr"],
            output_names=["image", "orig_size"],
            do_constant_folding=True,
        )
        export_args["dynamic_axes"] = {"image_bgr": {0: "h", 1: "w"}}

        torch.onnx.export(
            preprocessor,
            img_tensor,
            preprocess_onnx_path,
            **export_args,
        )

        # Añadir bypass para places_db
        model = onnx.load(preprocess_onnx_path)
        graph = model.graph

        places_input = helper.make_tensor_value_info(
            name="places_db",
            elem_type=onnx.TensorProto.FLOAT,
            shape=[None, None],
        )
        graph.input.append(places_input)

        places_node = helper.make_node(
            "Identity",
            inputs=["places_db"],
            outputs=["places_db_out"],
            name="Identity_ExposePlacesPre",
        )
        graph.node.append(places_node)

        places_output = helper.make_tensor_value_info(
            name="places_db_out",
            elem_type=onnx.TensorProto.FLOAT,
            shape=[None, None],
        )
        graph.output.append(places_output)

        onnx.save(model, preprocess_onnx_path)

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

        # Cargar modelo en ONNX y exponer imagen y orig_size
        model = onnx.load(detector_onnx_path)
        graph = model.graph

        # Bypass image
        input_name = graph.input[0].name
        identity_node = helper.make_node(
            "Identity",
            inputs=[input_name],
            outputs=["images_out"],
            name="Identity_ExposeImages",
        )

        orig_node = helper.make_node(
            "Identity",
            inputs=["orig_size"],
            outputs=["orig_size_det_out"],
            name="Identity_ExposeOrigSizeDet",
        )

        graph.node.extend([identity_node, orig_node])

        input_type = graph.input[0].type.tensor_type.elem_type
        input_shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param for dim in graph.input[0].type.tensor_type.shape.dim]
        new_output = helper.make_tensor_value_info(
            name="images_out",
            elem_type=input_type,
            shape=input_shape,
        )
        graph.output.append(new_output)

        # Bypass places_db
        places_input = helper.make_tensor_value_info(
            name="places_db",
            elem_type=onnx.TensorProto.FLOAT,
            shape=[None, None],
        )
        graph.input.append(places_input)

        places_node = helper.make_node(
            "Identity",
            inputs=["places_db"],
            outputs=["places_db_det_out"],
            name="Identity_ExposePlacesDet",
        )
        graph.node.append(places_node)

        places_output = helper.make_tensor_value_info(
            name="places_db_det_out",
            elem_type=onnx.TensorProto.FLOAT,
            shape=[None, None],
        )
        graph.output.append(places_output)

        # Bypass orig_size
        orig_input = helper.make_tensor_value_info(
            name="orig_size",
            elem_type=onnx.TensorProto.FLOAT,
            shape=[2],
        )
        graph.input.append(orig_input)

        orig_output = helper.make_tensor_value_info(
            name="orig_size_det_out",
            elem_type=onnx.TensorProto.FLOAT,
            shape=[2],
        )
        graph.output.append(orig_output)

        onnx.save(model, detector_onnx_path)

        return detector_onnx_path

    def _export_extractor(self, extractor, extractor_onnx_path: str, test_image_path: str, places_db):
        image_proc, orig_size, places_db = self.preprocess(test_image_path, places_db)
        detections, image_proc, places_db, orig_size = self.detect(image_proc, places_db, orig_size)

        img_tensor = torch.from_numpy(image_proc)
        if isinstance(detections, (list, tuple)):
            detections = detections[0]
        detections_tensor = torch.from_numpy(detections)

        # Exportar a ONNX
        torch.onnx.export(
            extractor,
            (detections_tensor, img_tensor),
            extractor_onnx_path,
            opset_version=16, # OPS versión >= 11 para NMS
            input_names=["detections", "image"],          
            output_names=["boxes", "descriptors"],
            dynamic_axes={
                "boxes":        {0: "num_boxes"},
                "descriptors":  {0: "num_boxes"}
            },
            do_constant_folding=True
        )

        model = onnx.load(extractor_onnx_path)
        graph = model.graph
        
        # Bypass places_db
        places_input = helper.make_tensor_value_info(
            name="places_db",
            elem_type=onnx.TensorProto.FLOAT,
            shape=[None, None],
        )
        graph.input.append(places_input)

        places_node = helper.make_node(
            "Identity",
            inputs=["places_db"],
            outputs=["places_db_out"],
            name="Identity_ExposePlacesExt",
        )
        graph.node.append(places_node)

        places_output = helper.make_tensor_value_info(
            name="places_db_out",
            elem_type=onnx.TensorProto.FLOAT,
            shape=[None, None],
        )
        graph.output.append(places_output)

        # Añadir bypass para orig_size
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
            name="Identity_ExposeOrigSizeExt",
        )
        graph.node.append(orig_node)

        orig_output = helper.make_tensor_value_info(
            name="orig_size_out",
            elem_type=onnx.TensorProto.FLOAT,
            shape=[2],
        )
        graph.output.append(orig_output)

        onnx.save(model, extractor_onnx_path)

    def _export_searcher(self, searcher, searcher_onnx_path: str, test_image_path: str, places_db):
        image_proc, orig_size, places_db = self.preprocess(test_image_path, places_db)
        detections, image_proc, places_db, orig_size = self.detect(image_proc, places_db, orig_size)

        if isinstance(detections, (list, tuple)):
            detections = detections[0]

        boxes, descriptors, places_db, orig_size = self.extract(detections, image_proc, places_db, orig_size)

        boxes = torch.from_numpy(boxes)
        descriptors = torch.from_numpy(descriptors)
        places_db = torch.from_numpy(places_db)

        torch.onnx.export(
            searcher,
            (
                boxes,
                descriptors,
                places_db,
            ),
            searcher_onnx_path,
            opset_version=16,
            input_names=[
                "boxes",
                "descriptors",
                "places_db",
            ],
            output_names=["boxes_out", "scores", "classes"],
            dynamic_axes={
                "boxes": {0: "num_boxes"},
                "descriptors": {0: "num_boxes"},
                "places_db": {0: "db_size"},
                "boxes_out": {0: "num_boxes"},
                "scores": {0: "num_boxes"},
                "classes": {0: "num_boxes"},
            },
            do_constant_folding=True,
        )

        # Añadir bypass para orig_size
        model = onnx.load(searcher_onnx_path)
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
            name="Identity_ExposeOrigSizeExt",
        )
        graph.node.append(orig_node)

        orig_output = helper.make_tensor_value_info(
            name="orig_size_out",
            elem_type=onnx.TensorProto.FLOAT,
            shape=[2],
        )
        graph.output.append(orig_output)

        onnx.save(model, searcher_onnx_path)
        
    def _export_postprocess(self, postprocessor, postprocess_onnx_path: str, test_image_path: str, places_db):
        image_proc, orig_size, places_db = self.preprocess(test_image_path, places_db)
        detections, image_proc, places_db, orig_size = self.detect(image_proc, places_db, orig_size)

        if isinstance(detections, (list, tuple)):
            detections = detections[0]

        boxes, descriptors, places_db, orig_size = self.extract(detections, image_proc, places_db, orig_size)
        boxes_out, scores_out, classes_out, orig_size = self.search(boxes, descriptors, places_db, orig_size)

        boxes_out = torch.from_numpy(boxes_out)
        scores_out = torch.from_numpy(scores_out)
        classes_out = torch.from_numpy(classes_out)
        orig_size = torch.from_numpy(orig_size)

        torch.onnx.export(
            postprocessor,
            (
                boxes_out,
                scores_out,
                classes_out,
                orig_size,
            ),
            postprocess_onnx_path,
            opset_version=16,
            input_names=[
                "final_boxes",
                "final_scores",
                "final_classes",
                "orig_size",
            ],
            output_names=["boxes", "scores", "classes"],
            dynamic_axes={
                "final_boxes": {0: "num_boxes"},
                "final_scores": {0: "num_boxes"},
                "final_classes": {0: "num_boxes"},
            },
            do_constant_folding=True,
        )

    def _export_pipeline(
        self,
        preprocess_onnx_path: str,
        detector_onnx_path: str,
        extractor_onnx_path: str,
        searcher_onnx_path: str,
        postprocess_onnx_path: str,
        pipeline_onnx_path: str,
    ):
        preprocessor_onnx = onnx.load(preprocess_onnx_path)
        detector_onnx = onnx.load(detector_onnx_path)
        extractor_onnx = onnx.load(extractor_onnx_path)
        searcher_onnx = onnx.load(searcher_onnx_path)
        postprocess_onnx = onnx.load(postprocess_onnx_path)

        # Prefix graphs to avoid name collisions when merging
        detector_onnx = compose.add_prefix(detector_onnx, "det_")
        extractor_onnx = compose.add_prefix(extractor_onnx, "ext_")
        searcher_onnx = compose.add_prefix(searcher_onnx, "ser_")
        postprocess_onnx = compose.add_prefix(postprocess_onnx, "post_")

        # Preprocess -> Detector
        det_inputs = [i.name for i in detector_onnx.graph.input]
        pre_outputs = [o.name for o in preprocessor_onnx.graph.output]
        merged_pd = compose.merge_models(
            preprocessor_onnx,
            detector_onnx,
            io_map=[
                (pre_outputs[0], det_inputs[0]),
                (pre_outputs[1], det_inputs[2]),
                (pre_outputs[2], det_inputs[1]),
            ],
        )

        # Detector -> Extractor
        ext_inputs = [i.name for i in extractor_onnx.graph.input]
        det_outputs = [o.name for o in detector_onnx.graph.output]

        merged_pde = compose.merge_models(
            merged_pd,
            extractor_onnx,
            io_map=[
                (det_outputs[0], ext_inputs[0]),  # output0 -> detections
                (det_outputs[1], ext_inputs[1]),  # images_out -> image
                (det_outputs[2], ext_inputs[2]),  # orig_size_det_out -> orig_size
                (det_outputs[3], ext_inputs[3]),  # places_db_det_out -> places_db
            ],
        )

        # Extractor -> Searcher
        ext_outputs = [o.name for o in extractor_onnx.graph.output]
        ser_inputs = [i.name for i in searcher_onnx.graph.input]

        merged_pdes = compose.merge_models(
            merged_pde,
            searcher_onnx,
            io_map=[
                (ext_outputs[0], ser_inputs[0]),
                (ext_outputs[1], ser_inputs[1]),
                (ext_outputs[2], ser_inputs[2]),
                (ext_outputs[3], ser_inputs[3]),
            ],
        )

        # Searcher -> Postprocess
        ser_outputs = [o.name for o in searcher_onnx.graph.output]
        post_inputs = [i.name for i in postprocess_onnx.graph.input]

        merged_model = compose.merge_models(
            merged_pdes,
            postprocess_onnx,
            io_map=[
                (ser_outputs[0], post_inputs[0]),
                (ser_outputs[1], post_inputs[1]),
                (ser_outputs[2], post_inputs[2]),
                (ser_outputs[3], post_inputs[3]),
            ],
        )

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

    def build_image_database(
        self,
        image_folder: str,
        df_pickle_path: str,
        descriptor_pickle_path: str,
        force_rebuild: bool = False,
        save_every: int = 500,
        min_area: float = 0.0,
        min_sim: float = 0.0,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Construye o actualiza una base de datos de descriptores.

        Este método procesa cada imagen de ``image_folder`` pasando solo por
        las etapas de ``preprocess`` → ``detect`` → ``extract`` para obtener los
        *embeddings* sin realizar la etapa de búsqueda.
        """

        if not os.path.isdir(image_folder):
            raise FileNotFoundError(f"La ruta '{image_folder}' no es un directorio válido.")

        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        all_files = [
            f for f in os.listdir(image_folder)
            if os.path.splitext(f)[1].lower() in valid_ext
        ]

        columns = ["image_name", "bbox", "class_id", "confidence"]

        if not force_rebuild and os.path.isfile(df_pickle_path):
            df_result = pd.read_pickle(df_pickle_path)
            if not set(columns).issubset(df_result.columns):
                raise ValueError("El DataFrame cargado no contiene las columnas requeridas.")
        else:
            df_result = pd.DataFrame(columns=columns)

        if not force_rebuild and os.path.isfile(descriptor_pickle_path):
            with open(descriptor_pickle_path, "rb") as f:
                descriptors_final = pickle.load(f)
            if not isinstance(descriptors_final, np.ndarray):
                raise ValueError("El archivo de descriptores no contiene un numpy.ndarray.")
        else:
            descriptors_final = np.zeros((0, 0), dtype=np.float32)

        if not force_rebuild and len(df_result) != len(descriptors_final):
            raise ValueError("El número de filas en el DataFrame no coincide con el número de descriptores.")

        C = descriptors_final.shape[1] if descriptors_final.size else 0
        processed = set(df_result["image_name"].unique()) if not force_rebuild else set()
        images_to_process = [f for f in all_files if f not in processed]

        new_rows = []
        new_descriptors = []
        init_C = C == 0
        processed_since_save = 0

        dummy_db = np.zeros((0, C + 1), dtype=np.float32) if C else np.zeros((0, 1), dtype=np.float32)

        for img_name in tqdm(images_to_process, desc="Procesando imágenes"):
            img_path = os.path.join(image_folder, img_name)
            try:
                img_proc, orig_size, places_db = self.preprocess(img_path, dummy_db)
                det_out = self.detect(img_proc, places_db, orig_size)
                detections = det_out[0] if isinstance(det_out, (list, tuple)) else det_out
                img_proc = det_out[1]
                places_db = det_out[2]
                orig_size = det_out[3]
                ext_out = self.extract(detections, img_proc, places_db, orig_size)
                boxes = ext_out[0]
                descriptors = ext_out[1]
            except Exception as e:
                print(f"Error procesando {img_name}: {e}")
                continue

            boxes_np = np.asarray(boxes)
            descriptors_np = np.asarray(descriptors)

            scores_np = np.ones(len(boxes_np), dtype=np.float32)
            classes_np = np.zeros(len(boxes_np), dtype=np.int64)
            if len(classes_np) > 0:
                classes_np[0] = -1

            if min_area > 0 and len(boxes_np) > 0:
                img_area = (boxes_np[0, 2] - boxes_np[0, 0]) * (boxes_np[0, 3] - boxes_np[0, 1])
                areas = (boxes_np[:, 2] - boxes_np[:, 0]) * (boxes_np[:, 3] - boxes_np[:, 1])
                mask = areas / img_area >= min_area
                boxes_np = boxes_np[mask]
                scores_np = scores_np[mask]
                classes_np = classes_np[mask]
                descriptors_np = descriptors_np[mask]

            if min_sim > 0 and len(descriptors_np) > 0:
                desc_t = torch.as_tensor(descriptors_np)
                sims = torch.matmul(desc_t, desc_t.T)
                adj = (sims >= min_sim).numpy()
                visited = np.zeros(len(desc_t), dtype=bool)
                best_group = []
                for i in range(len(desc_t)):
                    if not visited[i]:
                        queue = [i]
                        visited[i] = True
                        group = []
                        while queue:
                            v = queue.pop(0)
                            group.append(v)
                            neighbors = np.where(adj[v])[0]
                            for n in neighbors:
                                if not visited[n]:
                                    visited[n] = True
                                    queue.append(n)
                        if len(group) > len(best_group):
                            best_group = group
                keep_idx = np.array(sorted(best_group), dtype=int)
                boxes_np = boxes_np[keep_idx]
                scores_np = scores_np[keep_idx]
                classes_np = classes_np[keep_idx]
                descriptors_np = descriptors_np[keep_idx]

            if boxes_np.shape[0] == 0:
                continue

            if init_C:
                C = descriptors_np.shape[1]
                descriptors_final = np.zeros((0, C), dtype=descriptors_np.dtype)
                dummy_db = np.zeros((0, C + 1), dtype=np.float32)
                init_C = False

            for j in range(boxes_np.shape[0]):
                row = {
                    "image_name": img_name,
                    "bbox": tuple(map(float, boxes_np[j].tolist())),
                    "class_id": int(classes_np[j]),
                    "confidence": float(scores_np[j]),
                }
                new_rows.append(row)
                new_descriptors.append(descriptors_np[j].reshape(1, C))

            processed_since_save += 1
            if processed_since_save >= save_every:
                if new_rows:
                    df_new = pd.DataFrame(new_rows)
                    if df_result.empty:
                        df_result = df_new
                    else:
                        df_result = pd.concat([df_result, df_new], ignore_index=True)
                    stacked = np.vstack(new_descriptors)
                    descriptors_final = np.vstack([descriptors_final, stacked]) if descriptors_final.size else stacked
                    new_rows = []
                    new_descriptors = []
                    pd.to_pickle(df_result, df_pickle_path)
                    with open(descriptor_pickle_path, "wb") as f:
                        pickle.dump(descriptors_final, f)
                processed_since_save = 0

        if new_rows:
            df_new = pd.DataFrame(new_rows)
            if df_result.empty:
                df_result = df_new
            else:
                df_result = pd.concat([df_result, df_new], ignore_index=True)
            stacked = np.vstack(new_descriptors)
            descriptors_final = np.vstack([descriptors_final, stacked]) if descriptors_final.size else stacked

        pd.to_pickle(df_result, df_pickle_path)
        with open(descriptor_pickle_path, "wb") as f:
            pickle.dump(descriptors_final, f)

        return df_result, descriptors_final

