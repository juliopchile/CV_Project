import os
from ultralytics import settings, YOLO
from config import backbones_directory, datasets_path

# Turn DVC false para que no moleste en el entrenamiento
settings.update({'dvc': False})


def testear_modelos(lista_de_model_pt: list[str], test_files: str = "test_files"):
    # Lista de imagenes
    lista_de_imagenes = []
    for imagen in os.listdir(test_files):
        lista_de_imagenes.append(os.path.join(test_files, imagen))

    # Testear cada modelo con cada imagen y guardar el resultado
    for best_pt in lista_de_model_pt:
        loaded_model = YOLO(best_pt)
        loaded_model.predict(source=lista_de_imagenes, save=True, name=best_pt, conf=0.3, iou=0.5, half=True)


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
    # Crear lista de modelos best.pt a usar
    lista_modelos_ale = ["DEEP_0001_SGD.pt", "DEEP_LO_DUP_L_SGD.pt", "SALMONS_LO_YOLOL_ADAM.pt", "SALMONS_YOLOL_SGD.pt",
                         "SALMONS_YOLOL_SGD_RETRAINED.pt"]
    lista_de_model_pt = [os.path.normpath(os.path.join(backbones_directory, modelo)) for modelo in lista_modelos_ale]

    # Lisa de los path de best.pt para testear
    for model in ["yolov9c-seg", "yolov9e-seg"]:
        for optimizer in ["Adam", "SGD", "Adam_finetuned", "SGD_finetuned"]:
            for dataset in datasets_path.keys():
                path = os.path.normpath(f"models/training/{model}/{dataset}/{optimizer}/weights/best.pt")
                if os.path.exists(path):
                    lista_de_model_pt.append(path)

    print(lista_de_model_pt)
    testear_modelos(lista_de_model_pt)

