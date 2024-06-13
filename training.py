from typing import Dict, Any


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


def get_training_params(model: str = "yolov8m-seg") -> Dict[str, Dict[str, Any]]:
    """
    Retorna un diccionario con diferentes configuraciones de entrenamiento para distintos modelos.

    Args:
        model (str): El nombre del modelo a utilizar en las rutas del proyecto.

    Returns:
        Dict[str, Dict[str, Any]]: Diccionario con configuraciones de entrenamiento.
    """
    salmones = dict(data="datasets/roboflow/Salmones/data.yaml", project=f"models/training/salmons/{model}")
    deepfish = dict(data="datasets/roboflow/Deepfish/data.yaml", project=f"models/training/deepfish/{model}")

    # Configuraciones específicas para cada combinación de modelo y optimizador
    salmons_sgd = salmones.copy()
    salmons_sgd.update({'epochs': 30, 'lr0': 0.0167, 'lrf': 0.4442, 'momentum': 0.7747, 'warmup_epochs': 0.9795,
                        'warmup_momentum': 0.1557, 'box': 0.0520, 'cls': 2.6108, 'name': "SGD"})
    deepfish_sgd = deepfish.copy()
    deepfish_sgd.update({'epochs': 30, 'lr0': 0.0061, 'lrf': 0.8425, 'momentum': 0.7695, 'warmup_epochs': 0.2328,
                         'warmup_momentum': 0.7853, 'box': 0.1923, 'cls': 1.8807, 'name': "SGD"})
    salmons_adam = salmones.copy()
    salmons_adam.update({'epochs': 70, 'lr0': 0.001, 'lrf': 0.01, 'momentum': 0.9156, 'warmup_epochs': 3.0199,
                         "weight_decay": 0.00049, 'warmup_momentum': 0.7969, 'box': 7.4559, 'cls': 0.3991, 'name': "Adam"})
    deepfish_adam = deepfish.copy()
    deepfish_adam.update({'epochs': 70, 'lr0': 0.001, 'lrf': 0.01, 'momentum': 0.9604, 'warmup_epochs': 2.805,
                          "weight_decay": 0.00068, 'warmup_momentum': 0.6484, 'box': 8.3705, 'cls': 0.3961, 'name': "Adam"})

    salmons_sgd_lo = salmones.copy()
    salmons_sgd_lo.update({'epochs': 30, 'lr0': 0.0167, 'lrf': 0.4442, 'momentum': 0.7747, 'warmup_epochs': 0.9795,
                           'warmup_momentum': 0.1558, 'box': 0.0520, 'cls': 2.6109, 'name': "SGD_LO"})
    deepfish_sgd_lo = deepfish.copy()
    deepfish_sgd_lo.update({'epochs': 30, 'lr0': 0.0044, 'lrf': 0.5773, 'momentum': 0.6495, 'warmup_epochs': 0.6872,
                            'warmup_momentum': 0.4875, 'box': 0.0884, 'cls': 0.2271, 'name': "SGD_LO"})
    salmons_adam_lo = salmones.copy()
    salmons_adam_lo.update({'epochs': 70, 'lr0': 0.001, 'lrf': 0.01, 'momentum': 0.9545, 'warmup_epochs': 2.7419,
                            "weight_decay": 0.00061, 'warmup_momentum': 0.7518, 'box': 7.3135, 'cls': 0.3774, 'name': "Adam_LO"})
    deepfish_adam_lo = deepfish.copy()
    deepfish_adam_lo.update({'epochs': 70, 'lr0': 0.001, 'lrf': 0.01, 'momentum': 0.9126, 'warmup_epochs': 2.6590,
                             "weight_decay": 0.00042, 'warmup_momentum': 0.6637, 'box': 8.6925, 'cls': 0.5561, 'name': "Adam_LO"})

    params = {
        "salmons_sgd": salmons_sgd, "deepfish_sgd": deepfish_sgd, "salmons_adam": salmons_adam,
        "deepfish_adam": deepfish_adam, "salmons_sgd_lo": salmons_sgd_lo, "deepfish_sgd_lo": deepfish_sgd_lo,
        "salmons_adam_lo": salmons_adam_lo, "deepfish_adam_lo": deepfish_adam_lo
    }

    return params


def get_training_params_shiny(model="yolov8m-seg"):
    shiny_v2_sgd = dict(epochs=30, lr0=0.0167, lrf=0.4442, momentum=0.7747, warmup_epochs=0.9795,
                        warmup_momentum=0.1557,
                        box=0.0520, cls=2.6108, project=f"models/training/shinyV2/{model}", name="SGD")
    shiny_v4_sgd = dict(epochs=30, lr0=0.0061, lrf=0.8425, momentum=0.7695, warmup_epochs=0.2328,
                        warmup_momentum=0.7853,
                        box=0.1923, cls=1.8807, project=f"models/training/shiny4/{model}", name="SGD")
    shiny_v2_adam = dict(epochs=70, lr0=0.001, lrf=0.01, momentum=0.6386, warmup_epochs=4.4933, warmup_momentum=0.7395,
                         box=0.1510, cls=0.4593, project=f"models/training/shinyV2/{model}", name="Adam")
    shiny_v4_adam = dict(epochs=70, lr0=0.001, lrf=0.01, momentum=0.9594, warmup_epochs=1.5076, warmup_momentum=0.7314,
                         box=0.1342, cls=0.9801, project=f"models/training/shiny4/{model}", name="Adam")

    shiny_v2_sgd_lo = dict(epochs=30, lr0=0.0167, lrf=0.4442, momentum=0.7747, warmup_epochs=0.9795,
                           warmup_momentum=0.1558,
                           box=0.0520, cls=2.6109, project=f"models/training/shinyV2/{model}", name="SGD_LO")
    shiny_v4_sgd_lo = dict(epochs=30, lr0=0.0044, lrf=0.5773, momentum=0.6495, warmup_epochs=0.6872,
                           warmup_momentum=0.4875,
                           box=0.0884, cls=0.2271, project=f"models/training/shiny4/{model}", name="SGD_LO")
    shiny_v2_adam_lo = dict(epochs=70, lr0=0.001, lrf=0.01, momentum=0.9555, warmup_epochs=4.5288,
                            warmup_momentum=0.2417,
                            box=0.1405, cls=0.4644, project=f"models/training/salmons/{model}", name="Adam_LO")
    shiny_v4_adam_lo = dict(epochs=70, lr0=0.001, lrf=0.01, momentum=0.6828, warmup_epochs=3.9584,
                            warmup_momentum=0.2615,
                            box=0.1672, cls=0.5641, project=f"models/training/shiny4/{model}", name="Adam_LO")

    params = {"salmons_sgd": shiny_v2_sgd, "deepfish_sgd": shiny_v4_sgd, "salmons_adam": shiny_v2_adam,
              "deepfish_adam": shiny_v4_adam, "salmons_sgd_lo": shiny_v2_sgd_lo, "deepfish_sgd_lo": shiny_v4_sgd_lo,
              "salmons_adam_lo": shiny_v2_adam_lo, "deepfish_adam_lo": shiny_v4_adam_lo}

    return params


if __name__ == "__main__":
    # Parámetros
    train_params = get_training_params()
    for k, v in train_params.items():
        print(k, v)
