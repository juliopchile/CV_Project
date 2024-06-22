import os
import copy
from typing import Dict, Any
from ultralytics import YOLO
from model_utils import model_path, load_ultralytics_YOLO
from config import datasets_path


def get_training_params_for_datasets(model: str = "yolov8x-seg") -> dict[str, dict[str, str]]:
    """
    Retorna un diccionario con diccionarios, cada uno son los hiperparámetros asociados a un dataset específico para un
    modelo de entrada. Los datasets son sacados de "datasets_path" del módulo config y los parámetros son por defecto.
    Args:
        model (str): El nombre del modelo a utilizar en las rutas del proyecto.

    Returns:
        Dict[str, Dict[str, Any]]: Diccionario con configuraciones de entrenamiento.
    """
    # Pesos a usar en el entrenamiento
    model_pt = model_path(model)

    # Crear hiperparámetros para cada dataset
    hiperparams = {}
    for dataset, path in datasets_path.items():
        hiperparams.update(
            {f"{dataset}": dict(data=path, project=f"models/training/{model}/{dataset}", model=model_pt)})

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
    :param model_pt_path:
    :param hyperparameters_dict:
    :return:
    """
    for value in hyperparameters_dict.values():
        local_model = load_ultralytics_YOLO(model_pt_path)
        local_model.train(**value)
        del local_model


def thread_safe_re_training(hyperparameters_dict: Dict[str, Dict[str, Any]]):
    for value in hyperparameters_dict.values():
        model_pt_path = value["model"]
        local_model = YOLO(model_pt_path)
        local_model.train(**value)
        del local_model


if __name__ == "__main__":
    """
    Experimento 1: lr0=0.01. Se congela el backbone por 100 epochs, luego finetune con lr0=0.001.
        Resultados: SGD y AdamW entrenan bien para Salmon. AdamW falló para Deepfish. Finetune del modelo E tarda mucho.
        Conclusiones: No volver a entrenar el modelo entero para tamaño E, AdamW debe entrenarse con lr0 más pequeño.
    Experimento 2: Entrenar ahora con lr0=0.001.
        Resultados: Buena convergencia con ambos, 50 epochs para SGD y 70 para AdamW.
        Conclusiones: No es necesario entrenar tantas épocas si se tiene un buen learning rate
    Experimento 3: Entrenar con entrada 640 para ver si empeora o mejora.
        Resultados:
        Conclusiones:
    """
    epochs = 100        # Exp 1 y 2
    epochs_sgd = 50     # Exp 3
    epochs_adam = 70    # Exp 3
    finetune = 15       # Cambiar según se requiera
    lr0 = 0.001         # Cambiar según se requiera

    modelo_c = "yolov9c-seg"
    modelo_e = "yolov9e-seg"

    # Parámetros por defecto para cada dataset, para ambos tamaños de modelo
    train_params_c = get_training_params_for_datasets(modelo_c)
    train_params_e = get_training_params_for_datasets(modelo_e)
    add_extra_training_params(train_params_c,lr0=lr0, batch=8, imgsz=640, single_cls=True, cos_lr=True, plots=True)
    add_extra_training_params(train_params_e,lr0=lr0, batch=8, imgsz=640, single_cls=True, cos_lr=True, plots=True)

    # ADAM SIZE C
    # Transfer learning yolov9c con Adam
    yolov9c_adam = copy.deepcopy(train_params_c)
    add_extra_training_params(yolov9c_adam, optimizer="AdamW", name="Adam_3", epochs=epochs_adam, freeze=10)
    # Fine-tune
    yolov9c_adam_finetune = copy.deepcopy(train_params_c)
    add_extra_training_params(yolov9c_adam_finetune, optimizer="AdamW", name="Adam_finetuned_3", epochs=finetune, lr0=lr0/10)
    for key, value in yolov9c_adam.items():
        yolov9c_adam_finetune[key]["model"] = os.path.join(value["project"], value["name"], "weights", "best.pt")

    # SGD SIZE C
    # Transfer learning yolov9c con SGD
    yolov9c_sgd = copy.deepcopy(train_params_c)
    add_extra_training_params(yolov9c_sgd, optimizer="SGD", name="SGD_3", epochs=epochs_sgd, freeze=10)
    # Fine-tune
    yolov9c_sgd_finetune = copy.deepcopy(train_params_c)
    add_extra_training_params(yolov9c_sgd_finetune, optimizer="SGD", name="SGD_finetuned_3", epochs=finetune, lr0=lr0/10)
    for key, value in yolov9c_sgd.items():
        yolov9c_sgd_finetune[key]["model"] = os.path.join(value["project"], value["name"], "weights", "best.pt")

    # ADAM SIZE E
    # Transfer learning yolov9e con Adam
    yolov9e_adam = copy.deepcopy(train_params_e)
    add_extra_training_params(yolov9e_adam, optimizer="AdamW", name="Adam_3", epochs=epochs_adam, freeze=30)
    # Fine-tune
    yolov9e_adam_finetune = copy.deepcopy(train_params_e)
    add_extra_training_params(yolov9e_adam_finetune, optimizer="AdamW", name="Adam_finetuned_3", epochs=finetune, lr0=lr0/10)
    for key, value in yolov9e_adam.items():
        yolov9e_adam_finetune[key]["model"] = os.path.join(value["project"], value["name"], "weights", "best.pt")

    # SGD SIZE E
    # Transfer learning yolov9e con SGD
    yolov9e_sgd = copy.deepcopy(train_params_e)
    add_extra_training_params(yolov9e_sgd, optimizer="SGD", name="SGD_3", epochs=epochs_sgd, freeze=30)
    # Fine-tune
    yolov9e_sgd_finetune = copy.deepcopy(train_params_e)
    add_extra_training_params(yolov9e_sgd_finetune, optimizer="SGD", name="SGD_finetuned_3", epochs=finetune, lr0=lr0/10)
    for key, value in yolov9e_sgd.items():
        yolov9e_sgd_finetune[key]["model"] = os.path.join(value["project"], value["name"], "weights", "best.pt")

    # Training en serie
    thread_safe_training(modelo_c, yolov9c_sgd)  # SGD Modelo Large
    thread_safe_training(modelo_c, yolov9c_adam)  # Adam Modelo Large
    thread_safe_training(modelo_e, yolov9e_sgd)  # SGD Modelo X-Large
    thread_safe_training(modelo_e, yolov9e_adam)  # Adam Modelo X-Large

    # Fine-tune (cargar best.pt de los entrenamientos anteriores y entrenarlos un poco más)
    thread_safe_re_training(yolov9c_sgd_finetune)  # SGD Modelo Large
    thread_safe_re_training(yolov9c_adam_finetune)  # Adam Modelo Large
    # thread_safe_re_training(yolov9e_sgd_finetune)  # SGD Modelo X-Large
    # thread_safe_re_training(yolov9e_adam_finetune)  # Adam Modelo X-Large

    # Training en paralelo (No recomendado a menos que tengas pc de la NASA)
    #try:
    #    # Train todos los casos
    #    thread1 = Thread(target=thread_safe_training, args=(modelo_c, yolov9c_sgd))
    #    thread1.start() # SGD Modelo Large
    #    thread2 = Thread(target=thread_safe_training, args=(modelo_c, yolov9c_adam))
    #    thread2.start() # Adam Modelo Large
    #    thread3 = Thread(target=thread_safe_training, args=(modelo_e, yolov9e_sgd))
    #    thread3.start() # SGD Modelo X-Large
    #    thread4 = Thread(target=thread_safe_training, args=(modelo_e, yolov9e_adam))
    #    thread4.start() # Adam Modelo X-Large
#
    #    # Esperar a que terminen
    #    thread1.join()
    #    thread2.join()
    #    thread3.join()
    #    thread4.join()
    #except Exception as e:
    #    print(f"An error occurred: {e}")
    #else:
    #    # Re-train todos los casos
    #    thread1_re = Thread(target=thread_safe_training, args=(modelo_c, yolov9c_sgd_finetune))
    #    thread1_re.start() # SGD Modelo Large
    #    thread2_re = Thread(target=thread_safe_training, args=(modelo_c, yolov9c_adam_finetune))
    #    thread2_re.start() # Adam Modelo Large
    #    thread3_re = Thread(target=thread_safe_training, args=(modelo_e, yolov9e_sgd_finetune))
    #    thread3_re.start() # SGD Modelo X-Large
    #    thread4_re = Thread(target=thread_safe_training, args=(modelo_e, yolov9e_adam_finetune))
    #    thread4_re.start() # Adam Modelo X-Large
#
    #    # Esperar a que terminen
    #    thread1_re.join()
    #    thread2_re.join()
    #    thread3_re.join()
    #    thread4_re.join()
