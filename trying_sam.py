import os
import sys
import cv2
from matplotlib import pyplot as plt
from ultralytics import YOLO, SAM, FastSAM
import numba
import numpy as np
from PIL import Image 
from ultralytics.models.sam import Predictor as SAMPredictor
from CustomFastSAMPrompt import CustomFastSAMPrompt
import torch


@numba.jit(nopython=True)
def calcular_centros_numba(bounding_boxes):
    if bounding_boxes.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int32)
    
    centros = np.empty((len(bounding_boxes), 2), dtype=np.int32)
    for i in range(len(bounding_boxes)):
        xmin, ymin, xmax, ymax = bounding_boxes[i]
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        centros[i, 0] = int(round(x_center))
        centros[i, 1] = int(round(y_center))
    return centros


# Cargar los modelos
model_sam_1 = SAM("models/backbone/sam_l.pt")
model_sam_2 = SAM("models/backbone/sam_l.pt")
model_mobile_sam_1 = SAM("models/backbone/mobile_sam.pt")
model_mobile_sam_2 = SAM("models/backbone/mobile_sam.pt")
model_fastsam = FastSAM("models/backbone/FastSAM-x.pt")
model_det = YOLO("models/backbone/SALMONS_YOLOL_SGD_RETRAINED.pt")

for image in os.listdir("test_files_salmon"):
    # Cargar imagen
    #im = Image.open(os.path.join("test_files", image)).resize(size=(640,640))

    # Hacer detección (se usa el modelo de segmentación entrenado, pero solo se usan sus bounding boxes)
    detection_results = model_det.predict(source=os.path.join("test_files", image), imgsz=640, conf=0.6, iou=0.3, save=True)
    
    # Obtener puntos de interes si existe detección
    if (detection_results[0].boxes.shape)[0] != 0:
        detection = True
        bounding_boxes = np.array([bb.cpu().numpy() for bb in detection_results[0].boxes.xyxy], dtype=np.float32)
        centers = calcular_centros_numba(bounding_boxes).tolist()
    else:
        detection = False
    
    # Hacer segmentación con SAM-b (sin usar caja englobante)
    model_sam_1(source=os.path.join("test_files", image), imgsz=640, conf=0.4, iou=0.9, save=True)
    
    # Hacer segmentación con Mobile-SAM (sin usar caja englobante)
    model_mobile_sam_1(source=os.path.join("test_files", image), imgsz=640, conf=0.4, iou=0.9, save=True)
    
    # Hacer segmentación con FastSAM (sin usar caja englobante)
    segmentation_results_fastsam = model_fastsam.predict(source=os.path.join("test_files", image), imgsz=640, conf=0.4, iou=0.9, save=True)
    
    if detection:
        # Hacer segmentación con SAM-b (usando caja englobante)
        model_sam_2(source=os.path.join("test_files", image), bboxes=bounding_boxes, imgsz=640, conf=0.4, iou=0.9, save=True)

        # Hacer segmentación con Mobile-SAM (usando caja englobante)
        model_mobile_sam_2(source=os.path.join("test_files", image), bboxes=bounding_boxes, imgsz=640, conf=0.4, iou=0.9, save=True)
        
        try:
            # Cargar la clase prompt FastSAM
            prompt_process_fastsam = CustomFastSAMPrompt(image=os.path.join("test_files", image), results=segmentation_results_fastsam, device="cuda")
            ann = prompt_process_fastsam.box_prompt(bboxes=bounding_boxes)
            prompt_process_fastsam.plot(annotations=ann, bboxes=bounding_boxes, output_path=f"./runs/prompt/{image}")


        except Exception as e:
            print(e)
    