import os
import copy
from typing import Dict, Any
from ultralytics import YOLO
from model_utils import model_path, load_ultralytics_YOLO
from config import datasets_path_seg, datasets_path_det


def get_training_params_for_datasets(model: str = "yolov8x-seg", segmentation: bool = True) -> dict[str, dict[str, str]]:
    """
    Retorna un diccionario con diccionarios, cada uno son los hiperparámetros asociados a un dataset específico para un
    modelo de entrada. Los datasets son sacados de "datasets_path" del módulo config y los parámetros son por defecto.
    Args:
        model (str): El nombre del modelo a utilizar en las rutas del proyecto.
        segmentation (bool): Define el entrenamiento si es para segmentación o detección

    Returns:
        Dict[str, Dict[str, Any]]: Diccionario con configuraciones de entrenamiento.

    """
    # Pesos a usar en el entrenamiento
    model_pt = model_path(model)

    # Crear hiperparámetros para cada dataset
    hiperparams = {}
    dataset_path = datasets_path_seg if segmentation else datasets_path_det
    for dataset, path in dataset_path.items():
        hiperparams.update({f"{dataset}": dict(data=path, project=f"models/training/{model}/{dataset}", model=model_pt)})

    return hiperparams


def add_extra_training_params(train_params: Dict[str, Dict[str, Any]], **extra_params: Any) -> None:
    """
    Actualiza cada diccionario dentro de train_params con los valores proporcionados en extra_params.

    Args:
        train_params (Dict[str, Dict[str, Any]]): Diccionario principal que contiene otros diccionarios.
        **extra_params (Any): Parámetros adicionales a agregar a cada diccionario dentro de train_params.
    """
    for clave in train_params:
        if isinstance(train_params[clave], dict):
            train_params[clave].update(extra_params)


def thread_safe_training(model_pt_path: str, hyperparameters_dict: Dict[str, Dict[str, Any]]):
    """
    Permite el entrenamiento con un dataset usando un modelo local. Permite el uso entrenamiento por hilos.
    El modelo se carga a partir del path entregado.
    :param model_pt_path:
    :param hyperparameters_dict:
    :return:
    """
    for value in hyperparameters_dict.values():
        local_model = YOLO(model_pt_path)
        local_model.train(**value)
        del local_model


def thread_safe_re_training(hyperparameters_dict: Dict[str, Dict[str, Any]]):
    """
    Permite el re-entrenamiento (finetune) con un dataset usando un modelo local. Permite el entrenamiento por hilos.
    No se requiere entregar path para el dataset, pues este viene incluido en el diccionario de hiperparámetros.
    :param hyperparameters_dict:
    :return:
    """
    for value in hyperparameters_dict.values():
        model_pt_path = value["model"]
        local_model = YOLO(model_pt_path)
        local_model.train(**value)
        del local_model


if __name__ == "__main__":
    epochs = 100
    finetune = 30
    lr0 = 0.001
    
    # Entrenar deteccion con Yolov9
    modelo_c = "yolov9c"
    modelo_e = "yolov9e"

    # Parámetros por defecto para cada dataset, para ambos tamaños de modelo
    train_params_c = get_training_params_for_datasets(modelo_c)
    train_params_e = get_training_params_for_datasets(modelo_e)
    add_extra_training_params(train_params_c, lr0=lr0, batch=8, imgsz=640, single_cls=True, cos_lr=True, plots=True)
    add_extra_training_params(train_params_e, lr0=lr0, batch=8, imgsz=640, single_cls=True, cos_lr=True, plots=True)
