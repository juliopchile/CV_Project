from config import yolov8_segmentation_models
from model_utils import load_ultralytics_YOLO, model_path
from training import get_training_params, add_extra_training_params

if __name__ == "__main__":
    # Cargar Modelo
    model_name = yolov8_segmentation_models[4]
    modelo = load_ultralytics_YOLO(model_name)
    model_path = model_path(model_name)

    # Cargar Train parameters
    train_params = get_training_params(model_name)
    add_extra_training_params(train_params, batch=0.7, single_cls=True, plots=True, exist_ok=True, model=model_path)

    # Entrenar
    for key, value in train_params.items():
        if key in ["salmons_sgd", "deepfish_sgd", "salmons_adam", "deepfish_adam"]:
            modelo.train(**value)
