import os
import time

import cv2
import numba
import numpy as np
from PIL import Image
from ultralytics import YOLO, SAM, FastSAM

from CustomFastSAMPrompt import CustomFastSAMPrompt
from skimage.transform import resize


def resize_masks(masks, new_shape):
    N, _, _ = masks.shape
    resized_masks = np.zeros((N, new_shape[0], new_shape[1]), dtype=np.uint8)

    for i in range(N):
        # Reescalar la máscara y redondear los valores para mantenerlos como 0 o 1
        resized_mask = resize(masks[i], new_shape, order=0, preserve_range=True, anti_aliasing=False)
        resized_masks[i] = np.round(resized_mask).astype(np.uint8)

    return resized_masks


def overlay_mask_on_image(image, masks):
    # Crea una copia de la imagen original para superponer la máscara
    overlay = image.copy()

    # Recorre cada máscara en `masks` y superponla a la imagen
    for mask in masks:
        # Crea una imagen de 3 canales para la máscara con el color deseado (aquí se usa rojo)
        colored_mask = np.zeros_like(image)
        colored_mask[mask == 1] = [0, 0, 255]  # Rojo en BGR

        # Superpone la máscara coloreada en la imagen original
        overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0)

    return overlay


@numba.jit(nopython=True)
def promedio(numbers):
    return sum(numbers) / len(numbers) if numbers else 0


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


def compare_sam_models(dataset: str = "deepfish"):
    """
    Función para testear el uso de segmentación de dos etapas con Segment Anything Model SAM, en un conjunto de imágenes
    de prueba.
    :return: None
    """
    # Cargar los modelos
    model_sam_1 = SAM("models/backbone/sam_b.pt")
    model_sam_2 = SAM("models/backbone/sam_b.pt")
    model_mobile_sam_1 = SAM("models/backbone/mobile_sam.pt")
    model_mobile_sam_2 = SAM("models/backbone/mobile_sam.pt")
    model_fastsam = FastSAM("models/backbone/FastSAM-x.pt")
    if dataset == "deepfish":
        model_det = YOLO("models/training/yolov9c-seg/Deepfish/SGD_3/weights/best.pt", task="segment")
        images_path = "test_files_deepfish"
    elif dataset == "salmon":
        model_det = YOLO("models/training/yolov9e-seg/Salmones/SGD_2/weights/best.pt", task="segment")
        images_path = "test_files_salmon"
    else:
        model_det = YOLO("models/training/yolov9c-seg/Deepfish/SGD_3/weights/best.pt", task="segment")
        images_path = "test_files_deepfish"

    # Inicializar variables
    time_taken_sam = []
    time_taken_mobile_sam = []
    time_taken_fastsam = []
    time_taken_prompt_sam = []
    time_taken_prompt_mobile_sam = []
    time_taken_prompt_fastsam = []

    for image in os.listdir(images_path):
        # Cargar imagen
        image_path = os.path.join("test_files", image)

        # Hacer detección (se usa el modelo de segmentación entrenado, pero solo se usan sus bounding boxes)
        detection_results = model_det.predict(source=image_path, conf=0.3, iou=0.4, save=True)

        # Obtener puntos de interes si existe detección
        if detection_results[0].boxes.shape[0] != 0:
            detection = True
            bboxes = np.array([bb.cpu().numpy() for bb in detection_results[0].boxes.xyxy], dtype=np.int16)
        else:
            detection = False

        # Hacer segmentación con SAM-b (sin usar caja englobante)
        results1 = model_sam_1(source=image_path, conf=0.4, iou=0.85, save=True)
        print("SAM")

        # Hacer segmentación con Mobile-SAM (sin usar caja englobante)
        results2 = model_mobile_sam_1(source=image_path, conf=0.4, iou=0.85, save=True)
        print("MobileSAM")

        # Hacer segmentación con FastSAM (sin usar caja englobante)
        results3 = model_fastsam.predict(source=image_path, conf=0.4, iou=0.85, save=True)
        print("FastSAM")

        if detection:
            # Hacer segmentación con SAM-b (usando caja englobante)
            results4 = model_sam_2(source=image_path, bboxes=bboxes, conf=0.4, iou=0.85, save=True)
            print("SAM Prompt")

            # Hacer segmentación con Mobile-SAM (usando caja englobante)
            results5 = model_mobile_sam_2(source=image_path, bboxes=bboxes, conf=0.4, iou=0.85, save=True)
            print("MobileSAM Prompt")

            try:
                # Crear la clase prompt FastSAM
                start_time = time.perf_counter()
                prompt_process_fastsam = CustomFastSAMPrompt(image=image_path, results=results3, device="cuda")
                ann = prompt_process_fastsam.box_prompt(bboxes=bboxes)
                # Guardar imagen
                prompt_process_fastsam.plot(annotations=ann, bboxes=bboxes, output_path=f"./runs/prompt/{image}")
                # Tiempo que toma en realizar el prompting (incluyendo la creación de la salida plot()/plot_to_result())
                end_time = time.perf_counter()
                time_taken = (end_time - start_time) * 1000
                print(f"\n{time_taken:.6f} ms")
                print("FastSAM Prompt")

            except Exception as e:
                print(e)

        time_taken_sam.append(sum(results1[0].speed.values()))
        time_taken_mobile_sam.append(sum(results2[0].speed.values()))
        time_taken_fastsam.append(sum(results3[0].speed.values()))
        try:
            time_taken_prompt_sam.append(sum(results4[0].speed.values()))
            time_taken_prompt_mobile_sam.append(sum(results5[0].speed.values()))
            time_taken_prompt_fastsam.append(time_taken + sum(results3[0].speed.values()))
        except Exception as e:
            print(e)

    time_taken_sam = promedio(time_taken_sam)
    time_taken_mobile_sam = promedio(time_taken_mobile_sam)
    time_taken_fastsam = promedio(time_taken_fastsam)
    time_taken_prompt_sam = promedio(time_taken_prompt_sam)
    time_taken_prompt_mobile_sam = promedio(time_taken_prompt_mobile_sam)
    time_taken_prompt_fastsam = promedio(time_taken_prompt_fastsam)

    tiempos = (f"\n"
               f"time_taken_sam = {time_taken_sam}\n"
               f"time_taken_mobile_sam = {time_taken_mobile_sam}\n"
               f"time_taken_fastsam = {time_taken_fastsam}\n"
               f"time_taken_prompt_sam = {time_taken_prompt_sam}\n"
               f"time_taken_prompt_mobile_sam = {time_taken_prompt_mobile_sam}\n"
               f"time_taken_prompt_fastsam = {time_taken_prompt_fastsam}\n")

    print(tiempos)


def fast_sam_video_inference(det_model, seg_model, video_path, imgsz: int = 640):
    # Load the YOLOv9 model
    model_det = YOLO(det_model, task="detect")
    model_fastsam = FastSAM(seg_model)
    prompt_process_fastsam = None

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Verifica si se abrió correctamente
    if not cap.isOpened():
        print("Error al abrir el video.")
    else:
        # Obtiene el ancho y alto del video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Ancho: {width}, Alto: {height}")

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        # Ensure the frame is resized to a consistent size
        resized_frame = cv2.resize(frame, (imgsz, imgsz))

        if success:
            # Run YOLOv9 inference on the frame
            detection_results = model_det.predict(source=resized_frame, imgsz=imgsz, conf=0.25, iou=0.6, max_det=10)

            # Obtener segmentación si existe detección
            if detection_results[0].boxes.shape[0] != 0:
                start_time = time.perf_counter() * 1000
                segmentation_results_fastsam = model_fastsam.predict(source=resized_frame, imgsz=imgsz, conf=0.25, iou=0.9)
                inference_time = time.perf_counter() * 1000
                bounding_boxes = np.array([bb.cpu().numpy() for bb in detection_results[0].boxes.xyxy], dtype=np.int16)

                try:
                    # Crear la clase prompt FastSAM si no está creada
                    if prompt_process_fastsam is None:
                        prompt_process_fastsam = CustomFastSAMPrompt(image=resized_frame, results=segmentation_results_fastsam,
                                                                     device="gpu")
                    else:
                        # Reconfigurar la instancia existente
                        prompt_process_fastsam.img = resized_frame
                        prompt_process_fastsam.results = segmentation_results_fastsam

                    # Obtener las máscaras según las bounding boxes
                    #ann = resize_masks(prompt_process_fastsam.box_prompt(bboxes=bounding_boxes), (height, width))
                    #image_plus_mask = overlay_mask_on_image(image=frame, masks=ann)

                    # Use plot_to_result con las máscaras (Esta parte es la que se demora mucho)
                    ann = prompt_process_fastsam.box_prompt(bboxes=bounding_boxes)
                    annotated_frame_prompt = prompt_process_fastsam.plot_to_result(ann, better_quality=True,
                                                                                   retina=False, bboxes=bounding_boxes)

                    pos_time = time.perf_counter() * 1000

                    # Visualize the segmentation
                    cv2.imshow("Detection", detection_results[0].plot())
                    cv2.imshow("SAM Prompt", annotated_frame_prompt)
                    print(f"Inference: {inference_time - start_time} ms")
                    print(f"PostProcess: {pos_time - inference_time} ms")
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
    model_det = YOLO(det_model, task="detect")
    model_sam = SAM(seg_model)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv9 inference on the frame
            detection_results = model_det.predict(source=frame, conf=0.25, iou=0.7)

            # Obtener puntos de interés si existe detección
            if detection_results[0].boxes.shape[0] != 0:
                bounding_boxes = np.array([bb.cpu().numpy() for bb in detection_results[0].boxes.xyxy], dtype=np.int16)
                centers = calcular_centros_numba(bounding_boxes).tolist()

                # Hacer segmentación con SAM (usando caja englobante)
                #results = model_sam(source=frame, bboxes=bounding_boxes, conf=0.25, iou=0.9)
                results = model_sam(source=frame, bboxes=bounding_boxes, points=centers, imgsz=640, conf=0.4, iou=0.9)

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
    det_model = "models/cuantizado/yolov9e-deepfish-shinysalmon.engine"
    fast_sam_model = "models/cuantizado/FastSAM-x-salmon.engine"
    sam_model = "models/backbone/sam_b.pt"
    video_path = "test_videos/FISH_azul.avi"

    fast_sam_video_inference(det_model, fast_sam_model, video_path)
    sam_video_inference(det_model, sam_model, video_path)
    compare_sam_models("salmon")
