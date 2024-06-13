import os
from ultralytics import FastSAM, SAM, YOLO, NAS
from config import backbone_path, model_to_download


# ? Descargar modelos
def model_path(name: str) -> str:
    """
    Constructs the model path given the model name.

    Args:
        name (str): The name of the model.

    Returns:
        str: The full path to the model file.
    """
    return os.path.join(backbone_path, f"{name}.pt")


def download_models(models_to_download: list) -> dict:
    """
    Downloads and loads the models specified in the list. If no list is provided, all models are downloaded.

    Args:
        models_to_download (list, optional): List of model names to download. Defaults to None.

    Returns:
        dict: A dictionary with the model names as keys and the loaded model objects as values.
    """

    model_map = {
        'sam_l': 'sam_l',
        'sam_b': 'sam_b',
        'mobile_sam': 'mobile_sam',
        'FastSAM-s': 'FastSAM-s',
        'FastSAM-x': 'FastSAM-x',
        'yolov8n-seg': 'yolov8n-seg',
        'yolov8s-seg': 'yolov8s-seg',
        'yolov8m-seg': 'yolov8m-seg',
        'yolov8l-seg': 'yolov8l-seg',
        'yolov8x-seg': 'yolov8x-seg',
        'yolov9c-seg': 'yolov9c-seg',
        'yolov9e-seg': 'yolov9e-seg',
        'yolov8n': 'yolov8n',
        'yolov8s': 'yolov8s',
        'yolov8m': 'yolov8m',
        'yolov8l': 'yolov8l',
        'yolov8x': 'yolov8x',
        'yolo_nas_s': 'yolo_nas_s',
        'yolo_nas_m': 'yolo_nas_m',
        'yolo_nas_l': 'yolo_nas_l'
    }

    load_model = {
        'sam_l': lambda: SAM(model_path('sam_l')),
        'sam_b': lambda: SAM(model_path('sam_b')),
        'mobile_sam': lambda: SAM(model_path('mobile_sam')),
        'FastSAM-s': lambda: FastSAM(model_path('FastSAM-s')),
        'FastSAM-x': lambda: FastSAM(model_path('FastSAM-x')),
        'yolov8n-seg': lambda: YOLO(model_path('yolov8n-seg')),
        'yolov8s-seg': lambda: YOLO(model_path('yolov8s-seg')),
        'yolov8m-seg': lambda: YOLO(model_path('yolov8m-seg')),
        'yolov8l-seg': lambda: YOLO(model_path('yolov8l-seg')),
        'yolov8x-seg': lambda: YOLO(model_path('yolov8x-seg')),
        'yolov9c-seg': lambda: YOLO(model_path('yolov9c-seg')),
        'yolov9e-seg': lambda: YOLO(model_path('yolov9e-seg')),
        'yolov8n': lambda: YOLO(model_path('yolov8n')),
        'yolov8s': lambda: YOLO(model_path('yolov8s')),
        'yolov8m': lambda: YOLO(model_path('yolov8m')),
        'yolov8l': lambda: YOLO(model_path('yolov8l')),
        'yolov8x': lambda: YOLO(model_path('yolov8x')),
        #'yolo_nas_s': lambda: NAS(model_path_noext('yolo_nas_s')),      # Descargar a mano
        #'yolo_nas_m': lambda: NAS(model_path_noext('yolo_nas_m')),      # Descargar a mano
        #'yolo_nas_l': lambda: NAS(model_path_noext('yolo_nas_l'))       # Descargar a mano
    }

    loaded_models = {}
    for model_name in models_to_download:
        if model_name in model_map:
            model_key = model_map[model_name]
            loaded_models[model_key] = load_model[model_key]()
        else:
            print(f"Warning: Model name '{model_name}' is not recognized.")

    return loaded_models


# ? Cargar modelos
def load_ultralytics_YOLO(model_name='yolov8n-seg'):
    model = YOLO(model_path(model_name))
    return model


def load_ultralytics_SAM(model_name='sam_b'):
    model = SAM(model_path(model_name))
    return model


def load_ultralytics_FastSAM(model_name='FastSAM-s'):
    model = FastSAM(model_path(model_name))
    return model


def load_ultralytics_NAS(model_name='yolo_nas_s'):
    model = NAS(model_path(model_name))
    return model


if __name__ == "__main__":
    # Descargar todos los modelos
    #models_to_download = download_models(model_to_download)

    # Mostrar información de todos los modelos
    #for name, model in models_to_download.items():
    #    print(name)
    #    model.info()

    # Cargar un modelo y hacer inferencia con la camara
    model = load_ultralytics_NAS()
    model.predict(source=0, show=True, save=False)
