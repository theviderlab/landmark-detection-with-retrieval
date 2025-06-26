import torch
from landmark_detection.preprocess import PreprocessModule
from landmark_detection.extract import CVNet_SG
from landmark_detection.postprocess import PostprocessModule
from landmark_detection.search import Similarity_Search

from ultralytics import YOLO
import os
import cv2
from typing import List
import numpy as np
import json

import onnxruntime as ort
import onnx
from onnx import compose
from onnx import helper

class Pipeline_Yolo_CVNet_SG():

    def __init__(
        self,
        detector_file: str = "detector.onnx",
        extractor_onnx_file: str = "extractor.onnx",
        searcher_onnx_file: str = "searcher.onnx",
        pipeline_onnx_file: str = "pipeline.onnx",
        image_dim: tuple[int] = (640, 640),
        orig_size: tuple[int, int] | None = None,
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
        eps: float       = 1e-8,
        topk: int = 5,
        min_sim: float = 0.8,
        min_votes: float = 0.0,
        remove_inner_boxes: float | None = None,
        join_boxes: bool = False,
    ):
        self.image_dim = image_dim
        self.orig_size = orig_size
        self.preprocess_module = PreprocessModule(image_dim, orig_size)
        self.postprocess_module = PostprocessModule(image_dim)

        # Obtener el directorio donde está este archivo Python (el módulo)
        module_dir = os.path.dirname(os.path.abspath(__file__))

        ## DETECTOR
        if not os.path.isabs(detector_file) and not os.path.exists(detector_file):
            # Construir la ruta absoluta dentro de este módulo
            detector_path = os.path.join(module_dir, "models", detector_file)
        else:
            detector_path = detector_file

        detector_file_ext = os.path.splitext(detector_path)[-1].lower()
        # Si detector_file es .pt crea .onnx
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

        ## EXTRACTOR
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

        ## SEARCHER
        # Si no existe el archivo .onnx se crea
        if not os.path.isabs(searcher_onnx_file) and not os.path.exists(searcher_onnx_file):
            # Construir la ruta absoluta dentro de este módulo
            searcher_onnx_path = os.path.join(module_dir, "models", searcher_onnx_file)
        else:
            searcher_onnx_path = searcher_onnx_file

        searcher = Similarity_Search(
            topk = topk,
            min_sim = min_sim,
            min_votes = min_votes,
            remove_inner_boxes = remove_inner_boxes,
            join_boxes = join_boxes,
        ).eval()

        ## PIPELINE      
        # Construir la ruta absoluta a test_images/test.jpg dentro de este módulo
        test_image_path = os.path.join(module_dir, "test_images", "test.jpg")

        print('Creando versión ONNX del preprocess')
        preprocess_onnx_path = os.path.join(module_dir, "models", "preprocess.onnx")
        self._export_preprocess(self.preprocess_module, preprocess_onnx_path, test_image_path)

        print('Creando versión ONNX del extractor')
        self._export_extractor(extractor, extractor_onnx_path, test_image_path)

        print('Creando versión ONNX del searcher')
        self._export_searcher(searcher, searcher_onnx_path, test_image_path)

        print('Creando versión ONNX del postprocess')
        postprocess_onnx_path = os.path.join(module_dir, "models", "postprocess.onnx")
        self._export_postprocess(self.postprocess_module, postprocess_onnx_path)

        print('Creando versión ONNX del pipeline completo')
        # Si no existe el archivo .onnx se crea
        if not os.path.isabs(pipeline_onnx_file) and not os.path.exists(pipeline_onnx_file):
            # Construir la ruta absoluta dentro de este módulo
            pipeline_onnx_path = os.path.join(module_dir, "models", pipeline_onnx_file)
        else:
            pipeline_onnx_path = pipeline_onnx_file

        self._export_pipeline(
            preprocess_onnx_path,
            detector_onnx_path,
            extractor_onnx_path,
            searcher_onnx_path,
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
    
    def detect(self, image):

        # Preprocesar imagen
        img, orig_size = self.preprocess(image)

        # Obtener detecciones
        detector_inputs = {self.detector.get_inputs()[0].name: img}
        if len(self.detector.get_inputs()) > 1:
            detector_inputs[self.detector.get_inputs()[1].name] = orig_size
        detections = self.detector.run(None, detector_inputs)

        return detections, img, orig_size
    
    def extract(self, image, detections, orig_size=None):
        extractor_inputs = {"detections": detections, "image": image}
        if orig_size is not None and len(self.extractor.get_inputs()) > 2:
            extractor_inputs[self.extractor.get_inputs()[2].name] = orig_size
        return self.extractor.run(None, extractor_inputs)

    def search(self, results, orig_size=None):
        pass

    def postprocess(self, results, orig_size):
        """Escala las cajas al tamaño original de la imagen."""
        results = list(results)
        if len(results) > 0:
            boxes = torch.as_tensor(results[0])
            scores = torch.as_tensor(results[1])
            classes = torch.as_tensor(results[2])
            descriptors = torch.as_tensor(results[3])
            orig = torch.tensor([
                orig_size[0],
                orig_size[1],
            ], dtype=torch.float32)
            (
                results[0],
                results[1],
                results[2],
                results[3],
            ) = self.postprocess_module(boxes, scores, classes, descriptors, orig)
            results = [r for r in results]
        return results
    
    def run(self, image):
        """Ejecuta la inferencia completa empleando el modelo ONNX unido."""
        if isinstance(image, str):
            img_bgr = cv2.imread(image)
            if img_bgr is None:
                raise FileNotFoundError(f"No se encontró la imagen en {image}")
        else:
            img_bgr = image

        orig_h, orig_w = img_bgr.shape[:2]

        if self.orig_size is not None:
            img_bgr = cv2.resize(img_bgr, self.orig_size)

        img_tensor = torch.as_tensor(img_bgr)

        pipeline_inputs = {self.pipeline.get_inputs()[0].name: img_tensor.numpy()}
        results = self.pipeline.run(None, pipeline_inputs)

        if self.orig_size is not None and len(results) > 0:
            boxes = torch.as_tensor(results[0])
            scale = torch.tensor(
                [
                    orig_w / float(self.orig_size[0]),
                    orig_h / float(self.orig_size[1]),
                    orig_w / float(self.orig_size[0]),
                    orig_h / float(self.orig_size[1]),
                ],
                dtype=boxes.dtype,
            )
            results = list(results)
            results[0] = (boxes * scale).numpy()

        return results
    
    def _export_preprocess(self, preprocess_module, preprocess_onnx_path: str, test_image_path: str):
        if self.orig_size is not None:
            img_tensor = torch.zeros(
                (self.orig_size[1], self.orig_size[0], 3), dtype=torch.uint8
            )
        else:
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
        if self.orig_size is None:
            export_args["dynamic_axes"] = {"image_bgr": {0: "h", 1: "w"}}

        torch.onnx.export(
            preprocess_module,
            img_tensor,
            preprocess_onnx_path,
            **export_args,
        )

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

    def _export_searcher(self, searcher, searcher_onnx_path: str, test_image_path: str):
        pass

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
        searcher_onnx_path: str,
        postprocess_onnx_path: str,
        pipeline_onnx_path: str,
    ):
        preprocess_onnx = onnx.load(preprocess_onnx_path)
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
        det_input_name = detector_onnx.graph.input[0].name
        det_orig_name = detector_onnx.graph.input[1].name
        merged_pd = compose.merge_models(
            preprocess_onnx,
            detector_onnx,
            io_map=[
                ("image", det_input_name),
                ("orig_size", det_orig_name),
            ],
        )

        # Detector -> Extractor
        ext_inputs = [i.name for i in extractor_onnx.graph.input]
        det_outputs = [o.name for o in detector_onnx.graph.output]
        merged_pde = compose.merge_models(
            merged_pd,
            extractor_onnx,
            io_map=[
                (det_outputs[1], ext_inputs[1]),  # images_out -> image
                (det_outputs[0], ext_inputs[0]),  # output0 -> detections
                (det_outputs[2], ext_inputs[2]),  # orig_size_det_out -> orig_size
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
                (ext_outputs[4], post_inputs[4]),
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


<<<<<<< Updated upstream
<<<<<<< Updated upstream
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

class Similarity_Search():
    """Realiza búsqueda de similitud y votación por mayoría para cada detección."""

    def __init__(
        self,
        topk: int = 5,
        min_sim: float = 0.8,
        min_votes: float = 0.0,
        remove_inner_boxes: float | None = None,
        join_boxes: bool = False,
    ) -> None:
        """Inicializa el buscador.

        Parameters
        ----------
        topk : int
            Número máximo de vecinos a considerar.
        min_sim : float
            Similitud mínima para aceptar un vecino.
        min_votes : float, optional
            Porcentaje mínimo de votos necesarios para asignar una clase.
        remove_inner_boxes : float, optional
            Umbral para descartar cajas más grandes que se solapan con otras
            más pequeñas del mismo ``landmark``. Si ``None`` no se aplica
            este filtrado.
        """

        if not 0.0 <= min_votes <= 1.0:
            raise ValueError("min_votes debe estar entre 0 y 1")

        self.topk = topk
        self.min_sim = min_sim
        self.min_votes = min_votes
        self.remove_inner_boxes = remove_inner_boxes
        self.join_boxes = join_boxes

    def __call__(
        self,
        final_boxes: torch.Tensor | np.ndarray,
        final_scores: torch.Tensor | np.ndarray,
        final_classes: torch.Tensor | np.ndarray,
        descriptors: torch.Tensor | np.ndarray,
        X: torch.Tensor,
        idx: torch.Tensor,
    ) -> tuple:
        """Asigna un ``landmark`` a cada detección mediante búsqueda de similitud.

        Parameters
        ----------
        final_boxes : torch.Tensor | numpy.ndarray
            Cajas detectadas por :meth:`Pipeline_Yolo_CVNet_SG.run`.
        final_scores : torch.Tensor | numpy.ndarray
            Confianza de detección (no utilizada).
        final_classes : torch.Tensor | numpy.ndarray
            Clases de detección (no utilizadas).
        descriptors : torch.Tensor | numpy.ndarray
            Descriptores de las detecciones de la consulta.
        X : torch.Tensor
            Descriptores de la base de datos con shape ``(N, C)``.
        idx : torch.Tensor
            Índice ``landmark`` asociado a cada fila de ``X`` con shape ``(N,)``.

        Returns
        -------
        tuple
            ``(boxes, sims, classes)`` tras la votación por mayoría.
        """

        Q = descriptors if isinstance(descriptors, torch.Tensor) else torch.tensor(descriptors)
        if Q.ndim != 2:
            raise ValueError("descriptors debe tener shape (D, C)")
        if X.ndim != 2:
            raise ValueError("X debe tener shape (N, C)")
        if X.shape[0] != len(idx):
            raise ValueError("idx debe tener la misma longitud que X")
        if Q.shape[1] != X.shape[1]:
            raise ValueError("Dimensión C de Q y X debe coincidir")

        sims = torch.matmul(Q, X.T)  # (D, N)
        top_sims, top_idx = torch.topk(sims, self.topk, dim=1)

        results: list[int | None] = []
        for sim_row, idx_row in zip(top_sims, top_idx):
            mask = sim_row >= self.min_sim
            if not torch.any(mask):
                results.append(None)
                continue
            places = idx[idx_row[mask]]
            unique_ids, counts = torch.unique(places, return_counts=True)
            majority_index = counts.argmax()
            majority = unique_ids[majority_index]
            vote_ratio = counts[majority_index].float() / mask.sum().float()
            if vote_ratio < self.min_votes:
                results.append(None)
                continue
            results.append(int(majority.item()))

        if isinstance(final_boxes, np.ndarray):
            boxes_tensor = torch.from_numpy(final_boxes)
        else:
            boxes_tensor = final_boxes

        if self.remove_inner_boxes is not None and len(results) > 1:
            results = self._remove_overlapping_boxes(boxes_tensor, results, self.remove_inner_boxes)

        # Obtener puntuación de similitud para la clase asignada
        sim_scores: list[float] = []
        for r, sim_row, idx_row in zip(results, top_sims, top_idx):
            if r is None:
                sim_scores.append(0.0)
                continue
            mask = idx[idx_row] == r
            if torch.any(mask):
                sim_scores.append(float(sim_row[mask].max().item()))
            else:
                sim_scores.append(0.0)

        boxes_out = boxes_tensor.detach().cpu().numpy() if isinstance(boxes_tensor, torch.Tensor) else np.asarray(boxes_tensor)
        scores_out = np.asarray(sim_scores, dtype=np.float32)
        classes_out = list(results)

        if self.join_boxes:
            boxes_out, scores_out, classes_out = self._join_boxes_by_class(boxes_out, scores_out, classes_out)

        return boxes_out, scores_out, classes_out

    def _remove_overlapping_boxes(
        self,
        boxes: torch.Tensor,
        labels: list[int | None],
        thr: float,
    ) -> list[int | None]:
        """Descarta las cajas grandes cuando se solapan demasiado con otras
        más pequeñas del mismo ``landmark``.

        El solape se calcula como ``intersección / área_menor``.
        """

        b = boxes.detach().cpu().numpy()
        areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        new_labels = list(labels)
        n = len(labels)
        for i in range(n):
            if new_labels[i] is None:
                continue
            for j in range(i + 1, n):
                if new_labels[j] is None or new_labels[i] != new_labels[j]:
                    continue
                x1 = max(b[i, 0], b[j, 0])
                y1 = max(b[i, 1], b[j, 1])
                x2 = min(b[i, 2], b[j, 2])
                y2 = min(b[i, 3], b[j, 3])
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                inter = w * h
                if inter == 0:
                    continue
                if areas[i] > areas[j]:
                    bigger, smaller = i, j
                else:
                    bigger, smaller = j, i
                if inter / areas[smaller] >= thr:
                    new_labels[bigger] = None
        return new_labels

    def _join_boxes_by_class(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: list[int | None],
    ) -> tuple[np.ndarray, np.ndarray, list[int | None]]:
        """Une las cajas de la misma clase en una sola que las englobe."""

        new_boxes: list[list[float]] = []
        new_scores: list[float] = []
        new_labels: list[int | None] = []

        processed: set[int] = set()
        for i, lbl in enumerate(labels):
            if lbl is None or i in processed:
                continue
            idxs = [j for j, l in enumerate(labels) if l == lbl]
            processed.update(idxs)
            cls_boxes = boxes[idxs]
            x1 = float(cls_boxes[:, 0].min())
            y1 = float(cls_boxes[:, 1].min())
            x2 = float(cls_boxes[:, 2].max())
            y2 = float(cls_boxes[:, 3].max())
            new_boxes.append([x1, y1, x2, y2])
            new_scores.append(float(scores[idxs].max()))
            new_labels.append(lbl)

        for i, lbl in enumerate(labels):
            if lbl is None:
                new_boxes.append(boxes[i].tolist())
                new_scores.append(float(scores[i]))
                new_labels.append(None)

        return np.asarray(new_boxes, dtype=boxes.dtype), np.asarray(new_scores, dtype=scores.dtype), new_labels
=======
>>>>>>> Stashed changes
>>>>>>> Stashed changes
