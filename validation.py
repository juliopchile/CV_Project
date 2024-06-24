import os

from ultralytics import YOLO
from config import datasets_path, backbones_directory


def get_dataset_from_weight_path(best_pt: str):
    best_pt = os.path.normpath(best_pt)
    # Dividir la ruta en partes
    parts = best_pt.split(os.sep)

    # Extraer los valores
    model = parts[2]
    dataset = parts[3]
    optimizer = parts[4]

    return model, dataset, optimizer


def get_dataset_path_from_alejandro_weight_pt(model_pt):
    if model_pt == r"models\backbone\DEEP_0001_SGD.pt":
        return "Deepfish"
    elif model_pt == r"models\backbone\DEEP_LO_DUP_L_SGD.pt":
        return "Deepfish"
    elif model_pt == r"models\backbone\SALMONS_LO_YOLOL_ADAM.pt":
        return "Salmones"
    elif model_pt == r"models\backbone\SALMONS_YOLOL_SGD.pt":
        return "Salmones"
    elif model_pt == r"models\backbone\SALMONS_YOLOL_SGD_RETRAINED.pt":
        return "Salmones"


if __name__ == "__main__":
    lista_modelos_ale = ["DEEP_0001_SGD.pt", "DEEP_LO_DUP_L_SGD.pt", "SALMONS_LO_YOLOL_ADAM.pt", "SALMONS_YOLOL_SGD.pt",
                         "SALMONS_YOLOL_SGD_RETRAINED.pt"]
    lista_de_model_pt = [os.path.normpath(os.path.join(backbones_directory, modelo)) for modelo in lista_modelos_ale]

    # Validar los modelos mios
    for best_path in lista_de_model_pt[5:]:
        model_name, dataset_name, optimizer = get_dataset_from_weight_path(best_path)
        validation_directory = f"{model_name}/{dataset_name}/{optimizer}"
        dataset_path = datasets_path[dataset_name]
        model = YOLO(best_path)
        model.val(data=dataset_path, imgsz=640, batch=9, conf=0.3, iou=0.5, max_det=30, save_json=True, rect=True)
        del model

    # Validar modelos entrenados del Alejandro
    for best_path in lista_de_model_pt[0:5]:
        dataset_name = get_dataset_path_from_alejandro_weight_pt(best_path)
        validation_directory, _ = os.path.splitext(best_path)
        dataset_path = datasets_path[dataset_name]
        model = YOLO(best_path)
        model.val(data=dataset_path, imgsz=640, batch=9, conf=0.3, iou=0.5, max_det=30, save_json=True, rect=True)
        del model

    # Validar uno por uno (a mano)
    model = YOLO("models/training/yolov9e-seg/ShinySalmonsV4/SGD/weights/best.pt")
    model.val(data=datasets_path["ShinySalmonsV4"], imgsz=640, batch=8, conf=0.3, iou=0.5, max_det=10, save_json=True)
    del model

    model = YOLO("models/training/yolov9c-seg/ShinySalmonsV4/Adam/weights/best.pt")
    model.val(data=datasets_path["ShinySalmonsV4"], imgsz=640, batch=8, conf=0.3, iou=0.5, max_det=10, save_json=True)
    del model
