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


def compare_sam_models():
    # Cargar los modelos
    model_sam_1 = SAM("models/backbone/sam_l.pt")
    model_sam_2 = SAM("models/backbone/sam_l.pt")
    model_mobile_sam_1 = SAM("models/backbone/mobile_sam.pt")
    model_mobile_sam_2 = SAM("models/backbone/mobile_sam.pt")
    model_fastsam = FastSAM("models/cuantizado/FastSAM-x-salmon.engine")
    model_det = YOLO("models/backbone/SALMONS_YOLOL_SGD_RETRAINED.pt", task="detect")

    for image in os.listdir("test_files_salmon"):
        # Cargar imagen
        im = Image.open(os.path.join("test_files", image)).resize(size=(640, 640))

        # Hacer detección (se usa el modelo de segmentación entrenado, pero solo se usan sus bounding boxes)
        detection_results_1 = model_det.predict(source=os.path.join("test_files", image), imgsz=640, conf=0.6, iou=0.3, save=True)
        detection_results_2 = model_det.predict(source=im, imgsz=640, conf=0.6, iou=0.3)

        # Obtener puntos de interes si existe detección
        if (detection_results_1[0].boxes.shape[0] != 0) and (detection_results_2[0].boxes.shape[0] != 0):
            detection = True
            bounding_boxes_1 = np.array([bb.cpu().numpy() for bb in detection_results_1[0].boxes.xyxy], dtype=np.int16)
            bounding_boxes_2 = np.array([bb.cpu().numpy() for bb in detection_results_2[0].boxes.xyxy], dtype=np.int16)
        else:
            detection = False

        # Hacer segmentación con SAM-b (sin usar caja englobante)
        model_sam_1(source=os.path.join("test_files", image), imgsz=640, conf=0.4, iou=0.9, save=True)

        # Hacer segmentación con Mobile-SAM (sin usar caja englobante)
        model_mobile_sam_1(source=os.path.join("test_files", image), imgsz=640, conf=0.4, iou=0.9, save=True)

        # Hacer segmentación con FastSAM (sin usar caja englobante)
        model_fastsam.predict(source=os.path.join("test_files", image), imgsz=640, conf=0.4, iou=0.9, save=True)
        segmentation_results_fastsam = model_fastsam.predict(source=im, imgsz=640, conf=0.4, iou=0.9)

        if detection:
            # Hacer segmentación con SAM-b (usando caja englobante)
            model_sam_2(source=os.path.join("test_files", image), bboxes=bounding_boxes_1, imgsz=640, conf=0.4, iou=0.9, save=True)

            # Hacer segmentación con Mobile-SAM (usando caja englobante)
            model_mobile_sam_2(source=os.path.join("test_files", image), bboxes=bounding_boxes_1, imgsz=640, conf=0.4, iou=0.9, save=True)

            try:
                pass
                # Cargar la clase prompt FastSAM
                prompt_process_fastsam = CustomFastSAMPrompt(image=im, results=segmentation_results_fastsam, device="cuda")
                ann = prompt_process_fastsam.box_prompt(bboxes=bounding_boxes_2)
                prompt_process_fastsam.plot(annotations=ann, bboxes=bounding_boxes_2, output_path=f"./runs/prompt/{image}")

            except Exception as e:
                print(e)


def fast_sam_video_inference(det_model, seg_model, video_path):
    # Load the YOLOv8 model
    model_det = YOLO(det_model, task="detect")
    model_fastsam = FastSAM(seg_model)
    prompt_process_fastsam = None

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        # Ensure the frame is resized to a consistent size
        resized_frame = cv2.resize(frame, (640, 640))

        if success:
            # Run YOLOv9 inference on the frame
            detection_results = model_det.predict(source=resized_frame, conf=0.4, iou=0.5, max_det=10)

            # Obtener puntos de interés si existe detección
            if detection_results[0].boxes.shape[0] != 0:
                segmentation_results_fastsam = model_fastsam.predict(source=resized_frame, conf=0.3, iou=0.9)
                bounding_boxes = np.array([bb.cpu().numpy() for bb in detection_results[0].boxes.xyxy], dtype=np.int16)

                try:
                    # Crear la clase prompt FastSAM si no está creada
                    if prompt_process_fastsam is None:
                        prompt_process_fastsam = CustomFastSAMPrompt(image=resized_frame, results=segmentation_results_fastsam, device="cuda")
                    else:
                        # Reconfigurar la instancia existente
                        prompt_process_fastsam.img = resized_frame
                        prompt_process_fastsam.results = segmentation_results_fastsam

                    # Obtener las máscaras según las bounding boxes
                    ann = prompt_process_fastsam.box_prompt(bboxes=bounding_boxes)

                    # Use plot_to_result con las máscaras
                    annotated_frame_prompt = prompt_process_fastsam.plot_to_result(ann, retina=True, bboxes=bounding_boxes)

                    # Visualize the segmentation
                    cv2.imshow("Detection", detection_results[0].plot())
                    cv2.imshow("SAM Prompt", annotated_frame_prompt)
                except Exception as e:
                    print(e)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()


def sam_video_inference(det_model, seg_model, video_path):
    # Load the model models
    model_det = YOLO(det_model)
    model_sam = SAM(seg_model)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        # Ensure the frame is resized to a consistent size
        resized_frame = cv2.resize(frame, (640, 640))

        if success:
            # Run YOLOv9 inference on the frame
            detection_results = model_det.predict(source=resized_frame, conf=0.5, iou=0.3)

            # Obtener puntos de interés si existe detección
            if detection_results[0].boxes.shape[0] != 0:
                bounding_boxes = np.array([bb.cpu().numpy() for bb in detection_results[0].boxes.xyxy], dtype=np.float32)
                #centers = calcular_centros_numba(bounding_boxes).tolist()

                # Hacer segmentación con SAM (usando caja englobante)
                results = model_sam(source=resized_frame, bboxes=bounding_boxes, imgsz=640, conf=0.3, iou=0.9)
                #results = model_sam(source=resized_frame, points=centers, imgsz=640, conf=0.4, iou=0.9)

                cv2.imshow("Detection", detection_results[0].plot())
                cv2.imshow("SAM Prompt", results[0].plot())

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    det_model = "models/backbone/SALMONS_YOLOL_SGD_RETRAINED.pt"
    fast_sam_model = "models/cuantizado/FastSAM-x-salmon.engine"
    sam_model = "models/backbone/mobile_sam.pt"
    video_path = "test_videos/FISH_verde.avi"

    #fast_sam_video_inference(det_model, fast_sam_model, video_path)
    #sam_video_inference(det_model, sam_model, video_path)
    compare_sam_models()
