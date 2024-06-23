from ultralytics import YOLO
from config import datasets_path


if __name__ == "__main__":
    # Validar los modelos mios
    # for best_path in lista_de_model_pt[5:]:
    #    model_name, dataset_name, optimizer = get_dataset_from_weight_path(best_path)
    #    validation_directory = f"{model_name}/{dataset_name}/{optimizer}"
    #    dataset_path = datasets_path[dataset_name]
    #    model = YOLO(best_path)
    #    model.val(data=dataset_path, imgsz=640, batch=9, conf=0.3, iou=0.5, max_det=30, save_json=True, rect=True)
    #    del model

    # Validar modelos entrenados del Alejandro
    # for best_path in lista_de_model_pt[0:5]:
    #    dataset_name = get_dataset_path_from_alejandro_weight_pt(best_path)
    #    validation_directory, _ = os.path.splitext(best_path)
    #    dataset_path = datasets_path[dataset_name]
    #    model = YOLO(best_path)
    #    model.val(data=dataset_path, imgsz=640, batch=9, conf=0.3, iou=0.5, max_det=30, save_json=True, rect=True)
    #    del model

    # Validar uno por uno
    model = YOLO("models/training/yolov9e-seg/ShinySalmonsV4/SGD/weights/best.pt")
    model.val(data=datasets_path["ShinySalmonsV4"], imgsz=640, batch=8, conf=0.3, iou=0.5, max_det=10, save_json=True)
    del model

    model = YOLO("models/training/yolov9c-seg/ShinySalmonsV4/Adam/weights/best.pt")
    model.val(data=datasets_path["ShinySalmonsV4"], imgsz=640, batch=8, conf=0.3, iou=0.5, max_det=10, save_json=True)
    del model