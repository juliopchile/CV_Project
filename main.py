import os
from ultralytics import settings, YOLO
from config import backbones_directory, datasets_path

# Turn DVC false para que no moleste en el entrenamiento
settings.update({'dvc': False})


def testear_modelos_imagenes(lista_de_model_pt: list[str], test_files: str = "test_files"):
    # Lista de imagenes
    lista_de_imagenes = []
    for imagen in os.listdir(test_files):
        lista_de_imagenes.append(os.path.join(test_files, imagen))

    # Testear cada modelo con cada imagen y guardar el resultado
    for best_pt in lista_de_model_pt:
        loaded_model = YOLO(best_pt, task="segment")
        loaded_model.predict(source=lista_de_imagenes, save=True, name=best_pt, conf=0.3, iou=0.5)


def testear_modelos_videos(lista_de_model_pt: list[str], test_files: str = "test_files"):
    # Lista de imagenes
    lista_de_videos = []
    for video in os.listdir(test_files):
        lista_de_videos.append(os.path.join(test_files, video))

    # Testear cada modelo con cada imagen y guardar el resultado
    for best_pt in lista_de_model_pt:
        for video in lista_de_videos:
            loaded_model = YOLO(best_pt, task="segment")
            loaded_model.predict(source=video, save=True, name=best_pt, conf=0.3, iou=0.5)


if __name__ == "__main__":
    # Crear lista de modelos Shiny_SalmonV4_1024.pt a usar
    #lista_modelos_ale = ["DEEP_0001_SGD.pt", "DEEP_LO_DUP_L_SGD.pt", "SALMONS_LO_YOLOL_ADAM.pt", "SALMONS_YOLOL_SGD.pt",
    #                     "SALMONS_YOLOL_SGD_RETRAINED.pt"]
    #lista_de_model_pt = [os.path.normpath(os.path.join(backbones_directory, modelo)) for modelo in lista_modelos_ale]

    # Lisa de los path de Shiny_SalmonV4_1024.pt para testear
    #for model in ["yolov9c-seg", "yolov9e-seg"]:
    #    for optimizer in ["Adam", "SGD", "Adam_finetuned", "SGD_finetuned"]:
    #        for dataset in datasets_path.keys():
    #            path = os.path.normpath(f"models/training/{model}/{dataset}/{optimizer}/weights/Shiny_SalmonV4_1024.pt")
    #            if os.path.exists(path):
    #                lista_de_model_pt.append(path)

    #testear_modelos(lista_de_model_pt)

    # Testear modelos Shiny
    test_videos_path = "test_videos"
    lista_de_modelos_shiny = ["models/backbone/SALMONS_YOLOL_SGD_RETRAINED.pt"]
    lista_de_modelos_shiny = []

    # Lista de los path a los weights para shiny models
    for weight_path in os.listdir("shiny_salmon_models/tensor_rt"):
        if "1024" not in weight_path:
            lista_de_modelos_shiny.append(os.path.join("shiny_salmon_models/tensor_rt", weight_path))

    testear_modelos_videos(lista_de_modelos_shiny, test_videos_path)
