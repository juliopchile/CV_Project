import os
import copy
from threading import Thread
from training import get_training_params_for_datasets, add_extra_training_params, thread_safe_training, \
    thread_safe_re_training


# ! Segmentación
# ? Definir los hiperparámetros de los experimentos

def experimento_1(modelo_c="yolov9c-seg", modelo_e="yolov9e-seg"):
    """
    Experimento 1: lr0=0.01. Se congela el backbone por 100 epochs, luego finetune de 30 epochs con lr0=0.001.
        Resultados: SGD y AdamW entrenan bien para Salmon. AdamW falló para Deepfish. Finetune del modelo E tarda mucho.
        Conclusiones: No volver a entrenar el modelo entero para tamaño E, AdamW debe entrenarse con lr0 más pequeño.
    """
    epochs = 100
    finetune = 30
    lr0 = 0.01

    modelo_c = "yolov9c-seg"
    modelo_e = "yolov9e-seg"

    # Parámetros por defecto para cada dataset, para ambos tamaños de modelo
    train_params_c = get_training_params_for_datasets(modelo_c)
    train_params_e = get_training_params_for_datasets(modelo_e)
    add_extra_training_params(train_params_c, lr0=lr0, batch=8, imgsz=1024, single_cls=True, cos_lr=True, plots=True)
    add_extra_training_params(train_params_e, lr0=lr0, batch=8, imgsz=1024, single_cls=True, cos_lr=True, plots=True)

    # ADAM SIZE C
    # Transfer learning yolov9c con Adam
    yolov9c_adam = copy.deepcopy(train_params_c)
    add_extra_training_params(yolov9c_adam, optimizer="AdamW", name="Adam", epochs=epochs, freeze=10)
    # Fine-tune
    yolov9c_adam_finetune = copy.deepcopy(train_params_c)
    add_extra_training_params(yolov9c_adam_finetune, optimizer="AdamW", name="Adam_finetuned", epochs=finetune,
                              lr0=(lr0 / 10))
    for key, value in yolov9c_adam.items():
        yolov9c_adam_finetune[key]["model"] = os.path.join(value["project"], value["name"], "weights", "best.pt")

    # SGD SIZE C
    # Transfer learning yolov9c con SGD
    yolov9c_sgd = copy.deepcopy(train_params_c)
    add_extra_training_params(yolov9c_sgd, optimizer="SGD", name="SGD", epochs=epochs, freeze=10)
    # Fine-tune
    yolov9c_sgd_finetune = copy.deepcopy(train_params_c)
    add_extra_training_params(yolov9c_sgd_finetune, optimizer="SGD", name="SGD_finetuned", epochs=finetune,
                              lr0=(lr0 / 10))
    for key, value in yolov9c_sgd.items():
        yolov9c_sgd_finetune[key]["model"] = os.path.join(value["project"], value["name"], "weights", "best.pt")

    # ADAM SIZE E
    # Transfer learning yolov9e con Adam
    yolov9e_adam = copy.deepcopy(train_params_e)
    add_extra_training_params(yolov9e_adam, optimizer="AdamW", name="Adam", epochs=epochs, freeze=30)
    # Fine-tune
    yolov9e_adam_finetune = copy.deepcopy(train_params_e)
    add_extra_training_params(yolov9e_adam_finetune, optimizer="AdamW", name="Adam_finetuned", epochs=finetune,
                              lr0=(lr0 / 10))
    for key, value in yolov9e_adam.items():
        yolov9e_adam_finetune[key]["model"] = os.path.join(value["project"], value["name"], "weights", "best.pt")

    # SGD SIZE E
    # Transfer learning yolov9e con SGD
    yolov9e_sgd = copy.deepcopy(train_params_e)
    add_extra_training_params(yolov9e_sgd, optimizer="SGD", name="SGD", epochs=epochs, freeze=30)
    # Fine-tune
    yolov9e_sgd_finetune = copy.deepcopy(train_params_e)
    add_extra_training_params(yolov9e_sgd_finetune, optimizer="SGD", name="SGD_finetuned", epochs=finetune,
                              lr0=(lr0 / 10))
    for key, value in yolov9e_sgd.items():
        yolov9e_sgd_finetune[key]["model"] = os.path.join(value["project"], value["name"], "weights", "best.pt")


def experimento_2(modelo_c="yolov9c-seg", modelo_e="yolov9e-seg"):
    """
    Experimento 2: Entrenar ahora con lr0=0.001.
        Resultados: Buena convergencia con ambos, 50 epochs para SGD y 70 para AdamW.
        Conclusiones: No es necesario entrenar tantas épocas si se tiene un buen learning rate
    """
    epochs = 100
    finetune = 30
    lr0 = 0.001

    # Parámetros por defecto para cada dataset, para ambos tamaños de modelo
    train_params_c = get_training_params_for_datasets(modelo_c)
    train_params_e = get_training_params_for_datasets(modelo_e)
    add_extra_training_params(train_params_c, lr0=lr0, batch=8, imgsz=1024, single_cls=True, cos_lr=True, plots=True)
    add_extra_training_params(train_params_e, lr0=lr0, batch=8, imgsz=1024, single_cls=True, cos_lr=True, plots=True)

    # ADAM SIZE C
    # Transfer learning yolov9c con Adam
    yolov9c_adam = copy.deepcopy(train_params_c)
    add_extra_training_params(yolov9c_adam, optimizer="AdamW", name="Adam_2", epochs=epochs, freeze=10)
    # Fine-tune
    yolov9c_adam_finetune = copy.deepcopy(train_params_c)
    add_extra_training_params(yolov9c_adam_finetune, optimizer="AdamW", name="Adam_finetuned_2", epochs=finetune,
                              lr0=(lr0 / 10))
    for key, value in yolov9c_adam.items():
        yolov9c_adam_finetune[key]["model"] = os.path.join(value["project"], value["name"], "weights", "best.pt")

    # SGD SIZE C
    # Transfer learning yolov9c con SGD
    yolov9c_sgd = copy.deepcopy(train_params_c)
    add_extra_training_params(yolov9c_sgd, optimizer="SGD", name="SGD_2", epochs=epochs, freeze=10)
    # Fine-tune
    yolov9c_sgd_finetune = copy.deepcopy(train_params_c)
    add_extra_training_params(yolov9c_sgd_finetune, optimizer="SGD", name="SGD_finetuned_2", epochs=finetune,
                              lr0=(lr0 / 10))
    for key, value in yolov9c_sgd.items():
        yolov9c_sgd_finetune[key]["model"] = os.path.join(value["project"], value["name"], "weights", "best.pt")

    # ADAM SIZE E
    # Transfer learning yolov9e con Adam
    yolov9e_adam = copy.deepcopy(train_params_e)
    add_extra_training_params(yolov9e_adam, optimizer="AdamW", name="Adam_2", epochs=epochs, freeze=30)

    # SGD SIZE E
    # Transfer learning yolov9e con SGD
    yolov9e_sgd = copy.deepcopy(train_params_e)
    add_extra_training_params(yolov9e_sgd, optimizer="SGD", name="SGD_2", epochs=epochs, freeze=30)

    # Retornar los hiperparámetros de entrenamiento para el experimento RUN 1
    hiperparameteros_1 = (yolov9c_sgd, yolov9c_adam, yolov9e_sgd, yolov9e_adam)
    hiperparameteros_2 = (yolov9c_sgd_finetune, yolov9c_adam_finetune)

    return hiperparameteros_1, hiperparameteros_2


def experimento_3(modelo_c="yolov9c-seg", modelo_e="yolov9e-seg"):
    """
    Experimento 3: Entrenar con entrada 640 para ver si empeora o mejora.
        Resultados: El modelo mejora para Deepfish en todas las métricas, mejora en F1 pero empeora en map para Salmon.
        Conclusiones: Se considera que es una mejora entrenar en 640x640 pero principalmente para SGD con Deepfish.
    """
    epochs_sgd = 50
    epochs_adam = 70
    finetune = 15
    lr0 = 0.001

    # Parámetros por defecto para cada dataset, para ambos tamaños de modelo
    train_params_c = get_training_params_for_datasets(modelo_c)
    train_params_e = get_training_params_for_datasets(modelo_e)
    add_extra_training_params(train_params_c, lr0=lr0, batch=8, imgsz=640, single_cls=True, cos_lr=True, plots=True)
    add_extra_training_params(train_params_e, lr0=lr0, batch=8, imgsz=640, single_cls=True, cos_lr=True, plots=True)

    # ADAM SIZE C
    # Transfer learning yolov9c con Adam
    yolov9c_adam = copy.deepcopy(train_params_c)
    add_extra_training_params(yolov9c_adam, optimizer="AdamW", name="Adam_3", epochs=epochs_adam, freeze=10)
    # Fine-tune
    yolov9c_adam_finetune = copy.deepcopy(train_params_c)
    add_extra_training_params(yolov9c_adam_finetune, optimizer="AdamW", name="Adam_finetuned_3", epochs=finetune,
                              lr0=(lr0 / 10))
    for key, value in yolov9c_adam.items():
        yolov9c_adam_finetune[key]["model"] = os.path.join(value["project"], value["name"], "weights", "best.pt")

    # SGD SIZE C
    # Transfer learning yolov9c con SGD
    yolov9c_sgd = copy.deepcopy(train_params_c)
    add_extra_training_params(yolov9c_sgd, optimizer="SGD", name="SGD_3", epochs=epochs_sgd, freeze=10)
    # Fine-tune
    yolov9c_sgd_finetune = copy.deepcopy(train_params_c)
    add_extra_training_params(yolov9c_sgd_finetune, optimizer="SGD", name="SGD_finetuned_3", epochs=finetune,
                              lr0=(lr0 / 10))
    for key, value in yolov9c_sgd.items():
        yolov9c_sgd_finetune[key]["model"] = os.path.join(value["project"], value["name"], "weights", "best.pt")

    # ADAM SIZE E
    # Transfer learning yolov9e con Adam
    yolov9e_adam = copy.deepcopy(train_params_e)
    add_extra_training_params(yolov9e_adam, optimizer="AdamW", name="Adam_3", epochs=epochs_adam, freeze=30)

    # SGD SIZE E
    # Transfer learning yolov9e con SGD
    yolov9e_sgd = copy.deepcopy(train_params_e)
    add_extra_training_params(yolov9e_sgd, optimizer="SGD", name="SGD_3", epochs=epochs_sgd, freeze=30)

    # Retornar los hiperparámetros de entrenamiento para el experimento RUN 1
    hiperparameteros_1 = (yolov9c_sgd, yolov9c_adam, yolov9e_sgd, yolov9e_adam)
    hiperparameteros_2 = (yolov9c_sgd_finetune, yolov9c_adam_finetune)

    return hiperparameteros_1, hiperparameteros_2


# ? Llevar a cabo los experimentos

def run_1(multi: bool = False):
    modelo_c = "yolov9c-seg"
    modelo_e = "yolov9e-seg"

    hiperparameteros_1, hiperparameteros_2 = experimento_1(modelo_c, modelo_e)
    yolov9c_sgd, yolov9c_adam, yolov9e_sgd, yolov9e_adam = hiperparameteros_1
    yolov9c_sgd_finetune, yolov9c_adam_finetune, yolov9e_sgd_finetune, yolov9e_adam_finetune = hiperparameteros_2

    if multi:
        # Training en paralelo (No recomendado a menos que tengas PC de la NASA)
        try:
            # Train todos los casos
            thread1 = Thread(target=thread_safe_training, args=(modelo_c, yolov9c_sgd))
            thread1.start()  # SGD Modelo Common
            thread2 = Thread(target=thread_safe_training, args=(modelo_c, yolov9c_adam))
            thread2.start()  # Adam Modelo Common
            thread3 = Thread(target=thread_safe_training, args=(modelo_e, yolov9e_sgd))
            thread3.start()  # SGD Modelo Enlarged
            thread4 = Thread(target=thread_safe_training, args=(modelo_e, yolov9e_adam))
            thread4.start()  # Adam Modelo Enlarged

            # Esperar a que terminen
            thread1.join()
            thread2.join()
            thread3.join()
            thread4.join()
        except Exception as e:
            print(f"An error occurred: {e}")
        else:
            # Re-train todos los casos
            thread1_re = Thread(target=thread_safe_re_training, args=(yolov9c_sgd_finetune))
            thread1_re.start()  # SGD Modelo Common
            thread2_re = Thread(target=thread_safe_re_training, args=(yolov9c_adam_finetune))
            thread2_re.start()  # Adam Modelo Common
            thread3_re = Thread(target=thread_safe_re_training, args=(yolov9e_sgd_finetune))
            thread3_re.start()  # SGD Modelo Enlarged
            thread4_re = Thread(target=thread_safe_re_training, args=(yolov9e_adam_finetune))
            thread4_re.start()  # Adam Modelo Enlarged

            # Esperar a que terminen
            thread1_re.join()
            thread2_re.join()
            thread3_re.join()
            thread4_re.join()

    else:
        # Training en serie
        thread_safe_training(modelo_c, yolov9c_sgd)  # SGD Modelo Large
        thread_safe_training(modelo_c, yolov9c_adam)  # Adam Modelo Large
        thread_safe_training(modelo_e, yolov9e_sgd)  # SGD Modelo X-Large
        thread_safe_training(modelo_e, yolov9e_adam)  # Adam Modelo X-Large

        # Fine-tune (cargar best.pt de los entrenamientos anteriores y entrenarlos un poco más)
        thread_safe_re_training(yolov9c_sgd_finetune)  # SGD Modelo Large
        thread_safe_re_training(yolov9c_adam_finetune)  # Adam Modelo Large
        thread_safe_re_training(yolov9e_sgd_finetune)  # SGD Modelo X-Large
        #thread_safe_re_training(yolov9e_adam_finetune)  # Adam Modelo X-Large (no se llevó a cabo porque tarda mucho)


def run_2(multi: bool = False):
    modelo_c = "yolov9c-seg"
    modelo_e = "yolov9e-seg"

    hiperparameteros_1, hiperparameteros_2 = experimento_2(modelo_c, modelo_e)
    yolov9c_sgd, yolov9c_adam, yolov9e_sgd, yolov9e_adam = hiperparameteros_1
    yolov9c_sgd_finetune, yolov9c_adam_finetune = hiperparameteros_2

    if multi:
        # Training en paralelo (No recomendado a menos que tengas PC de la NASA)
        try:
            # Train todos los casos
            thread1 = Thread(target=thread_safe_training, args=(modelo_c, yolov9c_sgd))
            thread1.start()  # SGD Modelo Common
            thread2 = Thread(target=thread_safe_training, args=(modelo_c, yolov9c_adam))
            thread2.start()  # Adam Modelo Common
            thread3 = Thread(target=thread_safe_training, args=(modelo_e, yolov9e_sgd))
            thread3.start()  # SGD Modelo Enlarged
            thread4 = Thread(target=thread_safe_training, args=(modelo_e, yolov9e_adam))
            thread4.start()  # Adam Modelo Enlarged

            # Esperar a que terminen
            thread1.join()
            thread2.join()
            thread3.join()
            thread4.join()
        except Exception as e:
            print(f"An error occurred: {e}")
        else:
            # Re-train todos los casos
            thread1_re = Thread(target=thread_safe_re_training, args=(yolov9c_sgd_finetune))
            thread1_re.start()  # SGD Modelo Common
            thread2_re = Thread(target=thread_safe_re_training, args=(yolov9c_adam_finetune))
            thread2_re.start()  # Adam Modelo Common

            # Esperar a que terminen
            thread1_re.join()
            thread2_re.join()

    else:
        # Training en serie
        thread_safe_training(modelo_c, yolov9c_sgd)  # SGD Modelo Large
        thread_safe_training(modelo_c, yolov9c_adam)  # Adam Modelo Large
        thread_safe_training(modelo_e, yolov9e_sgd)  # SGD Modelo X-Large
        thread_safe_training(modelo_e, yolov9e_adam)  # Adam Modelo X-Large

        # Fine-tune (cargar best.pt de los entrenamientos anteriores y entrenarlos un poco más)
        thread_safe_re_training(yolov9c_sgd_finetune)  # SGD Modelo Large
        thread_safe_re_training(yolov9c_adam_finetune)  # Adam Modelo Large


def run_3(multi: bool = False):
    modelo_c = "yolov9c-seg"
    modelo_e = "yolov9e-seg"

    hiperparameteros_1, hiperparameteros_2 = experimento_3(modelo_c, modelo_e)
    yolov9c_sgd, yolov9c_adam, yolov9e_sgd, yolov9e_adam = hiperparameteros_1
    yolov9c_sgd_finetune, yolov9c_adam_finetune = hiperparameteros_2

    if multi:
        # Training en paralelo (No recomendado a menos que tengas PC de la NASA)
        try:
            # Train todos los casos
            thread1 = Thread(target=thread_safe_training, args=(modelo_c, yolov9c_sgd))
            thread1.start()  # SGD Modelo Common
            thread2 = Thread(target=thread_safe_training, args=(modelo_c, yolov9c_adam))
            thread2.start()  # Adam Modelo Common
            thread3 = Thread(target=thread_safe_training, args=(modelo_e, yolov9e_sgd))
            thread3.start()  # SGD Modelo Enlarged
            thread4 = Thread(target=thread_safe_training, args=(modelo_e, yolov9e_adam))
            thread4.start()  # Adam Modelo Enlarged

            # Esperar a que terminen
            thread1.join()
            thread2.join()
            thread3.join()
            thread4.join()
        except Exception as e:
            print(f"An error occurred: {e}")
        else:
            # Re-train todos los casos
            thread1_re = Thread(target=thread_safe_re_training, args=(yolov9c_sgd_finetune))
            thread1_re.start()  # SGD Modelo Common
            thread2_re = Thread(target=thread_safe_re_training, args=(yolov9c_adam_finetune))
            thread2_re.start()  # Adam Modelo Common

            # Esperar a que terminen
            thread1_re.join()
            thread2_re.join()

    else:
        # Training en serie
        thread_safe_training(modelo_c, yolov9c_sgd)  # SGD Modelo Large
        thread_safe_training(modelo_c, yolov9c_adam)  # Adam Modelo Large
        thread_safe_training(modelo_e, yolov9e_sgd)  # SGD Modelo X-Large
        thread_safe_training(modelo_e, yolov9e_adam)  # Adam Modelo X-Large

        # Fine-tune (cargar best.pt de los entrenamientos anteriores y entrenarlos un poco más)
        thread_safe_re_training(yolov9c_sgd_finetune)  # SGD Modelo Large
        thread_safe_re_training(yolov9c_adam_finetune)  # Adam Modelo Large


def train_shiny_salmons_seg(best_model_deepfish, best_model_salmones, modelo="yolov9c-seg"):
    lr0 = 0.001

    # Parámetros por defecto para cada dataset, para ambos tamaños de modelo
    train_params = get_training_params_for_datasets(modelo)
    add_extra_training_params(train_params, lr0=lr0, batch=8, imgsz=640, single_cls=True, cos_lr=True, plots=True)

    # Entrenar con el mejor modelo en Deepfish
    yolov9c_deepfish = copy.deepcopy(train_params)
    add_extra_training_params(yolov9c_deepfish, optimizer="SGD", name="Deepfish_SGD", epochs=70, freeze=10,
                              model=best_model_deepfish)
    # Fine-tune
    yolov9c_deepfish_finetune = copy.deepcopy(train_params)
    add_extra_training_params(yolov9c_deepfish_finetune, optimizer="SGD", name="Deepfish_SGD_finetuned", epochs=30,
                              lr0=lr0 / 10)
    for key, value in yolov9c_deepfish.items():
        yolov9c_deepfish_finetune[key]["model"] = os.path.join(value["project"], value["name"], "weights", "best.pt")

    # Entrenar con el mejor modelo en Salmones
    yolov9c_salmones = copy.deepcopy(train_params)
    add_extra_training_params(yolov9c_salmones, optimizer="SGD", name="Salmones_SGD", epochs=70, freeze=10,
                              model=best_model_salmones)
    # Fine-tune
    yolov9c_salmones_finetune = copy.deepcopy(train_params)
    add_extra_training_params(yolov9c_salmones_finetune, optimizer="SGD", name="Salmones_SGD_finetuned", epochs=30,
                              lr0=lr0 / 10)
    for key, value in yolov9c_salmones.items():
        yolov9c_salmones_finetune[key]["model"] = os.path.join(value["project"], value["name"], "weights", "best.pt")

    # Llevar a cabo los entrenamientos
    thread_safe_re_training(yolov9c_deepfish)
    thread_safe_re_training(yolov9c_deepfish_finetune)
    thread_safe_re_training(yolov9c_salmones)
    thread_safe_re_training(yolov9c_salmones_finetune)


# ! Detección
# ? Definir los hiperparámetros de los experimentos
def experimento_4(modelo_c="yolov9c", modelo_e="yolov9e"):
    """
    Experimento 4
    """
    epochs = 300
    finetune = 30
    lr0 = 0.001

    # Parámetros por defecto para cada dataset, para ambos tamaños de modelo
    train_params_c = get_training_params_for_datasets(modelo_c, False)
    train_params_e = get_training_params_for_datasets(modelo_e, False)
    add_extra_training_params(train_params_c, lr0=lr0, batch=8, imgsz=640, single_cls=True, cos_lr=True, plots=True,
                              patience=20)
    add_extra_training_params(train_params_e, lr0=lr0, batch=8, imgsz=640, single_cls=True, cos_lr=True, plots=True,
                              patience=20)

    # ADAM SIZE C
    # Transfer learning yolov9c con Adam
    yolov9c_adam = copy.deepcopy(train_params_c)
    add_extra_training_params(yolov9c_adam, optimizer="AdamW", name="Adam", epochs=epochs, freeze=10)
    # Fine-tune
    yolov9c_adam_finetune = copy.deepcopy(train_params_c)
    add_extra_training_params(yolov9c_adam_finetune, optimizer="AdamW", name="Adam_finetuned", epochs=finetune,
                              lr0=(lr0 / 10))
    for key, value in yolov9c_adam.items():
        yolov9c_adam_finetune[key]["model"] = os.path.join(value["project"], value["name"], "weights", "best.pt")

    # SGD SIZE C
    # Transfer learning yolov9c con SGD
    yolov9c_sgd = copy.deepcopy(train_params_c)
    add_extra_training_params(yolov9c_sgd, optimizer="SGD", name="SGD", epochs=epochs, freeze=10)
    # Fine-tune
    yolov9c_sgd_finetune = copy.deepcopy(train_params_c)
    add_extra_training_params(yolov9c_sgd_finetune, optimizer="SGD", name="SGD_finetuned", epochs=finetune,
                              lr0=(lr0 / 10))
    for key, value in yolov9c_sgd.items():
        yolov9c_sgd_finetune[key]["model"] = os.path.join(value["project"], value["name"], "weights", "best.pt")

    # ADAM SIZE E
    # Transfer learning yolov9e con Adam
    yolov9e_adam = copy.deepcopy(train_params_e)
    add_extra_training_params(yolov9e_adam, optimizer="AdamW", name="Adam", epochs=epochs, freeze=30)

    # SGD SIZE E
    # Transfer learning yolov9e con SGD
    yolov9e_sgd = copy.deepcopy(train_params_e)
    add_extra_training_params(yolov9e_sgd, optimizer="SGD", name="SGD", epochs=epochs, freeze=30)

    # Retornar los hiperparámetros de entrenamiento para el experimento RUN 1
    hiperparameteros_1 = (yolov9c_sgd, yolov9c_adam, yolov9e_sgd, yolov9e_adam)
    hiperparameteros_2 = (yolov9c_sgd_finetune, yolov9c_adam_finetune)

    return hiperparameteros_1, hiperparameteros_2


# ? Llevar a cabo los experimentos

def run_4(multi: bool = False):
    modelo_c = "yolov9c"
    modelo_e = "yolov9e"

    hiperparameteros_1, hiperparameteros_2 = experimento_4(modelo_c, modelo_e)
    yolov9c_sgd, yolov9c_adam, yolov9e_sgd, yolov9e_adam = hiperparameteros_1
    yolov9c_sgd_finetune, yolov9c_adam_finetune = hiperparameteros_2

    if multi:
        # Training en paralelo (No recomendado a menos que tengas PC de la NASA)
        try:
            # Train todos los casos
            thread1 = Thread(target=thread_safe_training, args=(modelo_c, yolov9c_sgd))
            thread1.start()  # SGD Modelo Common
            thread2 = Thread(target=thread_safe_training, args=(modelo_c, yolov9c_adam))
            thread2.start()  # Adam Modelo Common
            thread3 = Thread(target=thread_safe_training, args=(modelo_e, yolov9e_sgd))
            thread3.start()  # SGD Modelo Enlarged
            thread4 = Thread(target=thread_safe_training, args=(modelo_e, yolov9e_adam))
            thread4.start()  # Adam Modelo Enlarged

            # Esperar a que terminen
            thread1.join()
            thread2.join()
            thread3.join()
            thread4.join()
        except Exception as e:
            print(f"An error occurred: {e}")
        else:
            # Re-train todos los casos
            thread1_re = Thread(target=thread_safe_re_training, args=(yolov9c_sgd_finetune))
            thread1_re.start()  # SGD Modelo Common
            thread2_re = Thread(target=thread_safe_re_training, args=(yolov9c_adam_finetune))
            thread2_re.start()  # Adam Modelo Common

            # Esperar a que terminen
            thread1_re.join()
            thread2_re.join()

    else:
        # Training en serie
        thread_safe_training(modelo_c, yolov9c_sgd)  # SGD Modelo Large
        thread_safe_training(modelo_c, yolov9c_adam)  # Adam Modelo Large
        thread_safe_training(modelo_e, yolov9e_sgd)  # SGD Modelo X-Large
        thread_safe_training(modelo_e, yolov9e_adam)  # Adam Modelo X-Large

        # Fine-tune (cargar best.pt de los entrenamientos anteriores y entrenarlos un poco más)
        thread_safe_re_training(yolov9c_sgd_finetune)  # SGD Modelo Large
        thread_safe_re_training(yolov9c_adam_finetune)  # Adam Modelo Large


def train_shiny_salmons_det(best_model_c, best_model_e, modelo_c="yolov9c", modelo_e="yolov9e"):
    lr0 = 0.001  # Cambiar según se requiera
    epochs = 70
    #patience = 12

    # Parámetros por defecto para cada dataset, para ambos tamaños de modelo
    train_params_c = get_training_params_for_datasets(modelo_c, False)
    train_params_e = get_training_params_for_datasets(modelo_e, False)
    add_extra_training_params(train_params_c, lr0=lr0, batch=8, imgsz=640, single_cls=True, cos_lr=True, plots=True)
    add_extra_training_params(train_params_e, lr0=lr0, batch=8, imgsz=640, single_cls=True, cos_lr=True, plots=True)

    # ShinySalmon C
    yolov9c_shiny = copy.deepcopy(train_params_c)
    add_extra_training_params(yolov9c_shiny, optimizer="SGD", name="SGD", epochs=epochs, freeze=10, model=best_model_c)
    # Fine-tune
    yolov9c_shiny_finetune = copy.deepcopy(train_params_c)
    add_extra_training_params(yolov9c_shiny_finetune, optimizer="SGD", name="SGD_finetuned", epochs=25, lr0=lr0 / 10)
    for key, value in yolov9c_shiny.items():
        yolov9c_shiny_finetune[key]["model"] = os.path.join(value["project"], value["name"], "weights", "best.pt")

    # ShinySalmon E
    yolov9e_shiny = copy.deepcopy(train_params_e)
    add_extra_training_params(yolov9e_shiny, optimizer="SGD", name="SGD", epochs=epochs, freeze=30, model=best_model_e)
    # Re-train
    yolov9e_shiny_finetune = copy.deepcopy(train_params_e)
    add_extra_training_params(yolov9e_shiny_finetune, optimizer="SGD", name="SGD_retrain", freeze=30, epochs=25,
                              lr0=lr0 / 10)
    for key, value in yolov9e_shiny.items():
        yolov9e_shiny_finetune[key]["model"] = os.path.join(value["project"], value["name"], "weights", "best.pt")

    thread_safe_re_training(yolov9c_shiny)
    thread_safe_re_training(yolov9c_shiny_finetune)
    thread_safe_re_training(yolov9e_shiny)
    thread_safe_re_training(yolov9e_shiny_finetune)
