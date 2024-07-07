import os

from ultralytics import YOLO
from config import datasets_path_seg, datasets_path_det

val_dict_det_v9 = {
    # yolov9c - Deepfish
    "Run_1": {
        "model": "yolov9c",
        "dataset": "Deepfish",
        "case": "SGD",
        "weight_path": "models/training/yolov9c/Deepfish/SGD/weights/best.pt",
    },
    "Run_2": {
        "model": "yolov9c",
        "dataset": "Deepfish",
        "case": "SGD_finetuned",
        "weight_path": "models/training/yolov9c/Deepfish/SGD_finetuned/weights/best.pt",
    },
    "Run_3": {
        "model": "yolov9c",
        "dataset": "Deepfish",
        "case": "Adam",
        "weight_path": "models/training/yolov9c/Adam/SGD/weights/best.pt",
    },
    "Run_4": {
        "model": "yolov9c",
        "dataset": "Deepfish",
        "case": "Adam_finetuned",
        "weight_path": "models/training/yolov9c/Deepfish/Adam_finetuned/weights/best.pt",
    },
    # yolov9c - Salmones
    "Run_5": {
        "model": "yolov9c",
        "dataset": "Salmones",
        "case": "SGD",
        "weight_path": "models/training/yolov9c/Salmones/SGD/weights/best.pt",
    },
    "Run_6": {
        "model": "yolov9c",
        "dataset": "Salmones",
        "case": "SGD_finetuned",
        "weight_path": "models/training/yolov9c/Salmones/SGD_finetuned/weights/best.pt",
    },
    "Run_7": {
        "model": "yolov9c",
        "dataset": "Salmones",
        "case": "Adam",
        "weight_path": "models/training/yolov9c/Salmones/Adam/weights/best.pt",
    },
    "Run_8": {
        "model": "yolov9c",
        "dataset": "Salmones",
        "case": "Adam_finetuned",
        "weight_path": "models/training/yolov9c/Salmones/Adam_finetuned/weights/best.pt",
    },
    # yolov9e - Deepfish
    "Run_9": {
        "model": "yolov9e",
        "dataset": "Deepfish",
        "case": "SGD",
        "weight_path": "models/training/yolov9e/Deepfish/SGD/weights/best.pt",
    },
    "Run_10": {
        "model": "yolov9e",
        "dataset": "Deepfish",
        "case": "Adam",
        "weight_path": "models/training/yolov9e/Deepfish/Adam/weights/best.pt",
    },
    # yolov9e - Salmones
    "Run_11": {
        "model": "yolov9e",
        "dataset": "Salmones",
        "case": "SGD",
        "weight_path": "models/training/yolov9e/Salmones/SGD/weights/best.pt",
    },
    "Run_12": {
        "model": "yolov9e",
        "dataset": "Salmones",
        "case": "Adam",
        "weight_path": "models/training/yolov9e/Salmones/Adam/weights/best.pt",
    },
}


if __name__ == "__main__":
    # Validar modelos de detección YoloV9
    for run, data in val_dict_det_v9.items():
        model_to_val = data["weight_path"]
        dataset = datasets_path_det[data["dataset"]]
        save_dir = os.path.join("val", data["model"], data["dataset"], data["case"])

        # Validar uno por uno (a mano)
        model = YOLO(model_to_val, task="detect")
        model.val(data=dataset, imgsz=640, conf=0.5, iou=0.5, max_det=15, name=save_dir)
        del model

    
    
