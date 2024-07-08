import os
from ultralytics import FastSAM, SAM, YOLO, NAS
from config import backbones_directory, yolo_models, sam_models, fast_sam_models, nas_models, downloadable_models


def model_path(name: str) -> str:
    """
    Constructs the model path given the model name.

    Args:
        name (str): The name of the model.

    Returns:
        str: The full path to the model file.
    """
    return os.path.join(backbones_directory, f"{name}.pt")


# ? Descargar modelos
def download_models(models_to_download: list = None):
    """
    Downloads and loads the models specified in the list. If no list is provided, all models are downloaded.

    Args:
        models_to_download (list, optional): List of model names to download. Defaults to None.

    Returns:
        None
    """

    if models_to_download is None:
        models_to_download = yolo_models + sam_models + fast_sam_models + nas_models

    for model in models_to_download:
        if model in yolo_models:
            YOLO(model_path(model))
        elif model in sam_models:
            SAM(model_path(model))
        elif model in fast_sam_models:
            FastSAM(model_path(model))
        elif model in nas_models:
            NAS(model_path(model))


# ? Cargar modelos
def load_ultralytics_YOLO(model_name='yolov8n-seg'):
    return YOLO(model_path(model_name))


def load_ultralytics_SAM(model_name='sam_b'):
    return SAM(model_path(model_name))


def load_ultralytics_FastSAM(model_name='FastSAM-s'):
    return FastSAM(model_path(model_name))


def load_ultralytics_NAS(model_name='yolo_nas_s'):
    return NAS(model_path(model_name))


# ? Exportar modelos
def export_to_onnx(model_path: str, **extra_params):
    model = YOLO(model_path)
    model.export(format="onnx", **extra_params)


def export_to_tensor_rt(model, **extra_params):
    model.export(format="engine", **extra_params)


if __name__ == "__main__":
    #lista_modelos = ['yolov9c-seg', 'yolov9e-seg', 'yolov9c', 'yolov9e', 'sam_l', 'sam_b', 'mobile_sam', 'FastSAM-x',
    #                 'yolov10l', 'yolov10x']
    # Descargar modelos
    #download_models(lista_modelos)

    # Exportar un modelo
    model_pt_path = "models/backbone/SALMONS_YOLOL_SGD_RETRAINED.engine"
    model = YOLO(model_pt_path)
    #export_to_tensor_rt(model, half=False, int8=True, imgsz=640, data="dataset_yaml/deepfish.yaml")
    #export_to_tensor_rt(model, half=False, int8=True, imgsz=640, data="dataset_yaml/salmones.yaml")
    export_to_tensor_rt(model, half=False, int8=True, imgsz=640, data="dataset_yaml/shiny_salmons_v4.yaml")
    #model.predict(source=0, show=True)
