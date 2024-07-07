import os
from ultralytics import settings, YOLO
from config import backbones_directory, datasets_path_seg
from threading import Thread
from training_runs import run_1, run_2, run_3, train_shiny_salmons, run_4

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

    # ? Parte 1.5 (entrenar para shiny salmons)

    # Mejores modelos para los entrenamientos anteriores con el dataset Salmon
    #best_model_c = "models/training/yolov9c-seg/Deepfish/SGD_3/weights/best.pt"
    #best_model_e = "models/training/yolov9e-seg/Salmones/SGD_2/weights/best.pt"

    # Entrenar para el dataset shiny salmons
    #train_shiny_salmons()
    
    # ? Parte 2 (entrenar yolo de detección)
    run_4()
