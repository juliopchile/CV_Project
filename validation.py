import os

from ultralytics import YOLO
from config import datasets_path_seg, datasets_path_det

val_dict_det_v9 = {
    # yolov9c - Deepfish
    "Run_1": {
        "model": "yolov9c",
        "dataset": "Deepfish",
        "case": "SGD",
        "weight_path": "models/training/yolov9c/Deepfish/SGD/weights/best.pt",
    },
    "Run_2": {
        "model": "yolov9c",
        "dataset": "Deepfish",
        "case": "SGD_finetuned",
        "weight_path": "models/training/yolov9c/Deepfish/SGD_finetuned/weights/best.pt",
    },
    "Run_3": {
        "model": "yolov9c",
        "dataset": "Deepfish",
        "case": "Adam",
        "weight_path": "models/training/yolov9c/Adam/SGD/weights/best.pt",
    },
    "Run_4": {
        "model": "yolov9c",
        "dataset": "Deepfish",
        "case": "Adam_finetuned",
        "weight_path": "models/training/yolov9c/Deepfish/Adam_finetuned/weights/best.pt",
    },
    # yolov9c - Salmones
    "Run_5": {
        "model": "yolov9c",
        "dataset": "Salmones",
        "case": "SGD",
        "weight_path": "models/training/yolov9c/Salmones/SGD/weights/best.pt",
    },
    "Run_6": {
        "model": "yolov9c",
        "dataset": "Salmones",
        "case": "SGD_finetuned",
        "weight_path": "models/training/yolov9c/Salmones/SGD_finetuned/weights/best.pt",
    },
    "Run_7": {
        "model": "yolov9c",
        "dataset": "Salmones",
        "case": "Adam",
        "weight_path": "models/training/yolov9c/Salmones/Adam/weights/best.pt",
    },
    "Run_8": {
        "model": "yolov9c",
        "dataset": "Salmones",
        "case": "Adam_finetuned",
        "weight_path": "models/training/yolov9c/Salmones/Adam_finetuned/weights/best.pt",
    },
    # yolov9e - Deepfish
    "Run_9": {
        "model": "yolov9e",
        "dataset": "Deepfish",
        "case": "SGD",
        "weight_path": "models/training/yolov9e/Deepfish/SGD/weights/best.pt",
    },
    "Run_10": {
        "model": "yolov9e",
        "dataset": "Deepfish",
        "case": "Adam",
        "weight_path": "models/training/yolov9e/Deepfish/Adam/weights/best.pt",
    },
    # yolov9e - Salmones
    "Run_11": {
        "model": "yolov9e",
        "dataset": "Salmones",
        "case": "SGD",
        "weight_path": "models/training/yolov9e/Salmones/SGD/weights/best.pt",
    },
    "Run_12": {
        "model": "yolov9e",
        "dataset": "Salmones",
        "case": "Adam",
        "weight_path": "models/training/yolov9e/Salmones/Adam/weights/best.pt",
    },
}

val_dict_det_shiny = {
    # yolov9c - Salmon
    "Run_1": {
        "model": "yolov9c",
        "dataset": "ShinySalmonsV4",
        "case": "SGD",
        "weight_path": "models/training/yolov9c/ShinySalmonsV4/SGD/weights/best.pt"
    },
    "Run_2": {
        "model": "yolov9c",
        "dataset": "ShinySalmonsV4",
        "case": "SGD_finetuned",
        "weight_path": "models/training/yolov9c/ShinySalmonsV4/SGD_finetuned/weights/best.pt"
    },
    "Run_3": {
        "model": "yolov9e",
        "dataset": "ShinySalmonsV4",
        "case": "SGD",
        "weight_path": "models/training/yolov9e/ShinySalmonsV4/SGD/weights/best.pt"
    },
    "Run_4": {
        "model": "yolov9e",
        "dataset": "ShinySalmonsV4",
        "case": "SGD_retrain",
        "weight_path": "models/training/yolov9e/ShinySalmonsV4/SGD_retrain/weights/best.pt"
    }
}

val_dict_seg_cuantization_alejandro = {
    # Alejandro - Deepfish - SGD1
    "Run_1": {
        "model": "Alejandro",
        "dataset": "Deepfish",
        "case": "DEEP_0001_SGD",
        "weight_path": "models/backbone/DEEP_0001_SGD.pt"
    },
    "Run_2": {
        "model": "Alejandro",
        "dataset": "Deepfish",
        "case": "DEEP_0001_SGD_ENGINE",
        "weight_path": "models/cuantizado/DEEP_0001_SGD.engine"
    },
    # Alejandro - Deepfish - SGD2
    "Run_3": {
        "model": "Alejandro",
        "dataset": "Deepfish",
        "case": "DEEP_LO_DUP_L_SGD",
        "weight_path": "models/backbone/DEEP_LO_DUP_L_SGD.pt"
    },
    "Run_4": {
        "model": "Alejandro",
        "dataset": "Deepfish",
        "case": "DEEP_LO_DUP_L_SGD_ENGINE",
        "weight_path": "models/cuantizado/DEEP_LO_DUP_L_SGD.engine"
    },
    # Alejandro - Salmones - SGD
    "Run_5": {
        "model": "Alejandro",
        "dataset": "Salmones",
        "case": "SALMONS_YOLOL_SGD",
        "weight_path": "models/backbone/SALMONS_YOLOL_SGD.pt"
    },
    "Run_6": {
        "model": "Alejandro",
        "dataset": "Salmones",
        "case": "SALMONS_YOLOL_SGD_ENGINE",
        "weight_path": "models/cuantizado/SALMONS_YOLOL_SGD.engine"
    },
    # Alejandro - Salmones - ADAM
    "Run_7": {
        "model": "Alejandro",
        "dataset": "Salmones",
        "case": "SALMONS_LO_YOLOL_ADAM",
        "weight_path": "models/backbone/SALMONS_LO_YOLOL_ADAM.pt"
    },
    "Run_8": {
        "model": "Alejandro",
        "dataset": "Salmones",
        "case": "SALMONS_LO_YOLOL_ADAM_ENGINE",
        "weight_path": "models/cuantizado/SALMONS_LO_YOLOL_ADAM.engine"
    },
    # Alejandro - ShinySalmons - SGD
    "Run_9": {
        "model": "Alejandro",
        "dataset": "ShinySalmonsV4",
        "case": "SALMONS_YOLOL_SGD_RETRAINED",
        "weight_path": "models/backbone/SALMONS_YOLOL_SGD_RETRAINED.pt"
    },
    "Run_10": {
        "model": "Alejandro",
        "dataset": "ShinySalmonsV4",
        "case": "SALMONS_YOLOL_SGD_RETRAINED_ENGINE",
        "weight_path": "models/cuantizado/SALMONS_YOLOL_SGD_RETRAINED.engine"
    },
}

val_dict_seg_cuantization_mios = {
    # yolov9c-seg - Deepfish - SGD
    "Run_1": {
        "model": "My_best",
        "dataset": "Deepfish",
        "case": "SGD",
        "weight_path": "models/training/yolov9c-seg/Deepfish/SGD_finetuned_3/weights/best.pt"
    },
    "Run_2": {
        "model": "My_best",
        "dataset": "Deepfish",
        "case": "SGD_ENGINE",
        "weight_path": "models/training/yolov9c-seg/Deepfish/SGD_finetuned_3/weights/best.engine"
    },
    # yolov9c-seg - Deepfish - Adam
    "Run_3": {
        "model": "My_best",
        "dataset": "Deepfish",
        "case": "Adam",
        "weight_path": "models/training/yolov9c-seg/Deepfish/Adam_3/weights/best.pt"
    },
    "Run_4": {
        "model": "My_best",
        "dataset": "Deepfish",
        "case": "Adam_ENGINE",
        "weight_path": "models/training/yolov9c-seg/Deepfish/Adam_3/weights/best.engine"
    },
    # yolov9c-seg - Salmones - SGD
    "Run_5": {
        "model": "My_best",
        "dataset": "Salmones",
        "case": "SGD",
        "weight_path": "models/training/yolov9c-seg/Salmones/SGD_finetuned_3/weights/best.pt"
    },
    "Run_6": {
        "model": "My_best",
        "dataset": "Salmones",
        "case": "SGD_ENGINE",
        "weight_path": "models/training/yolov9c-seg/Salmones/SGD_finetuned_3/weights/best.engine"
    },
    # yolov9c-seg - Salmones - Adam
    "Run_7": {
        "model": "My_best",
        "dataset": "Salmones",
        "case": "Adam",
        "weight_path": "models/training/yolov9c-seg/Salmones/Adam_finetuned_3/weights/best.pt"
    },
    "Run_8": {
        "model": "My_best",
        "dataset": "Salmones",
        "case": "Adam_ENGINE",
        "weight_path": "models/training/yolov9c-seg/Salmones/Adam_finetuned_3/weights/best.engine"
    }
}

val_dict_seg_shiny = {
    # Pre-trained con Deepfish
    "Run_1": {
        "model": "My_best",
        "dataset": "ShinySalmonsV4",
        "case": "Deepfish_SGD",
        "weight_path": "models/training/yolov9c-seg/ShinySalmonsV4/Deepfish_SGD/weights/best.pt"
    },
    "Run_2": {
        "model": "My_best",
        "dataset": "ShinySalmonsV4",
        "case": "Deepfish_SGD_ENGINE",
        "weight_path": "models/training/yolov9c-seg/ShinySalmonsV4/Deepfish_SGD/weights/best.engine"
    },
    "Run_3": {
        "model": "My_best",
        "dataset": "ShinySalmonsV4",
        "case": "Deepfish_SGD_finetuned",
        "weight_path": "models/training/yolov9c-seg/ShinySalmonsV4/Deepfish_SGD_finetuned/weights/best.pt"
    },
    "Run_4": {
        "model": "My_best",
        "dataset": "ShinySalmonsV4",
        "case": "Deepfish_SGD_finetuned_ENGINE",
        "weight_path": "models/training/yolov9c-seg/ShinySalmonsV4/Deepfish_SGD_finetuned/weights/best.engine"
    },
    # Pre-trained con Salmones
    "Run_5": {
        "model": "My_best",
        "dataset": "ShinySalmonsV4",
        "case": "Salmones_SGD",
        "weight_path": "models/training/yolov9c-seg/ShinySalmonsV4/Salmones_SGD/weights/best.pt"
    },
    "Run_6": {
        "model": "My_best",
        "dataset": "ShinySalmonsV4",
        "case": "Salmones_SGD_ENGINE",
        "weight_path": "models/training/yolov9c-seg/ShinySalmonsV4/Salmones_SGD/weights/best.engine"
    },
    "Run_7": {
        "model": "My_best",
        "dataset": "ShinySalmonsV4",
        "case": "Salmones_SGD_finetuned",
        "weight_path": "models/training/yolov9c-seg/ShinySalmonsV4/Salmones_SGD_finetuned/weights/best.pt"
    },
    "Run_8": {
        "model": "My_best",
        "dataset": "ShinySalmonsV4",
        "case": "Salmones_SGD_finetuned_ENGINE",
        "weight_path": "models/training/yolov9c-seg/ShinySalmonsV4/Salmones_SGD_finetuned/weights/best.engine"
    }
}


def validar_experimentos(val_dict, datasets_path, task='segment'):
    for run, data in val_dict.items():
        model_to_val = data["weight_path"]
        dataset = datasets_path[data["dataset"]]
        save_dir = os.path.join("val", data["model"], data["dataset"], data["case"])

        # Validar uno por uno
        model = YOLO(model_to_val, task=task)
        model.val(data=dataset, name=save_dir)
        del model
        print()


if __name__ == "__main__":
    # Seleccionar diccionario de validación según el caso que se desee
    val_dict = val_dict_seg_shiny  # El run o experimento a validar
    datasets_path = datasets_path_seg  # Si es detección o segmentación

    # Validar modelos
    validar_experimentos(val_dict, datasets_path, 'segment')

