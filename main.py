import os
from ultralytics import settings, YOLO, FastSAM
from config import backbones_directory, datasets_path_seg
from threading import Thread
from training_runs import run_1, run_2, run_3, train_shiny_salmons_seg, run_4, train_shiny_salmons_det
from ultralytics.utils.benchmarks import benchmark

# Turn DVC false para que no moleste en el entrenamiento
settings.update({'dvc': False})


def testear_modelos_imagenes(lista_de_model_pt: list[str], lista_names: list[str], test_files: str = "test_files"):
    # Lista de imagenes
    lista_de_imagenes = []
    for imagen in os.listdir(test_files):
        lista_de_imagenes.append(os.path.join(test_files, imagen))

    # Testear cada modelo con cada imagen y guardar el resultado
    for best_pt, name in zip(lista_de_model_pt, lista_names):
        loaded_model = YOLO(best_pt, task="segment")
        loaded_model.predict(source=lista_de_imagenes, save=True, name=name, conf=0.3, iou=0.5)


def testear_modelos_videos(lista_de_model_pt: list[str], lista_names: list[str], test_files: str = "test_files"):
    # Lista de imagenes
    lista_de_videos = []
    for video in os.listdir(test_files):
        lista_de_videos.append(os.path.join(test_files, video))

    # Testear cada modelo con cada imagen y guardar el resultado
    for model, name in zip(lista_de_model_pt, lista_names):
        for video in lista_de_videos:
            t = Thread(target=thread_safe_predict, args=(model, video, name))
            t.start()
            t.join()


def thread_safe_predict(best_pt, video, name):
    """Performs thread-safe prediction on an image using a locally instantiated YOLO model."""
    loaded_model = YOLO(best_pt, task="segment")
    results = loaded_model.predict(source=video, save=True, name=name, conf=0.3, iou=0.5, stream=True)


if __name__ == "__main__":
    # ? Parte 1 (segmentación con yolov9-seg)
    """
    * Experimento 1: lr0=0.01. Se congela el backbone por 100 epochs, luego finetune con lr0=0.001.
    *     Resultados: SGD y AdamW entrenan bien para Salmon. AdamW falló para Deepfish. Finetune del modelo E tarda mucho.
    *     Conclusiones: No volver a entrenar el modelo entero para tamaño E, AdamW debe entrenarse con lr0 más pequeño.
    * Experimento 2: Entrenar ahora con lr0=0.001.
    *     Resultados: Buena convergencia con ambos, 50 epochs para SGD y 70 para AdamW.
    *     Conclusiones: No es necesario entrenar tantas épocas si se tiene un buen learning rate
    * Experimento 3: Entrenar con entrada 640 para ver si empeora o mejora.
    *     Resultados: El modelo mejora para Deepfish en todas las métricas, mejora en F1 pero empeora en map para Salmon.
    *     Conclusiones: Se considera que es una mejora entrenar en 640x640 pero principalmente para SGD con Deepfish.
    """

    # Descargar datasets (dataset_utils.py)

    # Llevar a cabo los entrenamientos para segmentación
    # run_1()
    # run_2()
    # run_3()

    # ? Parte 1.5 (entrenar segmentación para shiny salmons)
    # Mejores modelos para los entrenamientos anteriores con el dataset Salmon (tamaño 640x640 osea run 3)
    # best_model_deepfish = "models/training/yolov9c-seg/Deepfish/SGD_finetuned_3/weights/best.pt"
    # best_model_salmones = "models/training/yolov9c-seg/Salmones/SGD_finetuned_3/weights/best.pt"
    # train_shiny_salmons_seg(best_model_deepfish, best_model_salmones)

    # ? Parte 2 (entrenar yolo de detección)
    # run_4()

    # ? Parte 2.5 (entrenar detección para shiny salmons)
    # Mejores modelos para los entrenamientos anteriores para ambos datasets
    # best_model_salmones = "models/training/yolov9c/Salmones/SGD_finetuned/weights/best.pt"
    # best_model_deepfish = "models/training/yolov9e/Deepfish/SGD/weights/best.pt"
    # train_shiny_salmons_det(best_model_salmones, best_model_deepfish)

    # Parte 3 (Tracking usando modelos de segmentación de la párte 1)
    # Mejor modelo para el dataset de ShinySalmons
    best_model_mio_path = "models/training/yolov9c-seg/ShinySalmonsV4/Salmones_SGD/weights/best.pt"
    best_model_mio_engine = "models/training/yolov9c-seg/ShinySalmonsV4/Salmones_SGD/weights/best.engine"
    best_model_alejandro_path = "models/backbone/SALMONS_YOLOL_SGD_RETRAINED.pt"
    best_model_alejandro_engine = "models/cuantizado/SALMONS_YOLOL_SGD_RETRAINED.engine"
    fast_sam = "models/backbone/FastSAM-x.pt"
    fast_sam_engine = "models/cuantizado/FastSAM-x-salmon.engine"

    # Hacer tracking
    model = FastSAM(fast_sam)
    model.track(source="test_videos/FISH_verde.avi", conf=0.3, iou=0.5, save=True, stream_buffer=True, tracker="botsort.yaml")
    #model.track(source="test_videos/FISH_azul.avi", conf=0.1, iou=0.2, show=True)


