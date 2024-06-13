import os
from copy import copy

from roboflow.core.dataset import Dataset
from config import datasets
from roboflow import Roboflow
from supersecrets import API_KEY


def download_roboflow_dataset(workspace, project_id, version_number, model_format, location):
    try:
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace(the_workspace=workspace).project(project_id=project_id)
        version = project.version(version_number=version_number)
        return version.download(model_format=model_format, location=location, overwrite=False)
    except Exception as error:
        print(error)
        return None


def get_roboflow_dataset(name):
    """
    Retorna un objeto de la clase Roboflow Dataset, respectivo a un dataset previamente descargado.
    :param name: Nombre del dataset, debe ser uno incluido en el diccionario "datasets"
    :return:
    """
    version = datasets[name]["version"]
    model_format = datasets[name]["model_format"]
    location = datasets[name]["location"]
    return Dataset(name, version, model_format, location)


def lista_txt_con_clase(dataset_path: str):
    labels_con_peces = []
    for carpeta in os.listdir(dataset_path):
        if carpeta in ["test", "train", "valid"]:
            labels_path = os.path.join(dataset_path, carpeta, "labels")
            if os.path.isdir(labels_path):
                for label_file in os.listdir(labels_path):
                    label_file_path = os.path.join(labels_path, label_file)
                    if os.path.isfile(label_file_path):
                        with open(label_file_path, 'r') as file:
                            content = file.read().strip()
                            if content:  # Check if the file content is not empty
                                labels_con_peces.append(label_file_path)

    return labels_con_peces


def lista_img_con_clase(labels_con_peces: list[str]):
    imagenes_con_peces = []
    for imagen in labels_con_peces:
        # Replace 'labels' with 'images'
        new_path = imagen.replace('labels', 'images')
        # Replace '.txt' extension with '.jpg'
        new_path = new_path.replace('.txt', '.jpg')
        imagenes_con_peces.append(new_path)

    return imagenes_con_peces


#def copiar_dataset(input_dataset: str, output_dataset: str):
#    input_dataset = "datasets/roboflow/Deepfish"
#    output_dataset = "datasets/roboflow/Deepfish_LO"
#    for carpeta in os.listdir(input_dataset):
#        if file not in ["test", "train", "valid"]:
#            labels_path = os.path.join(dataset_path, carpeta, "labels")
#            if os.path.isdir(labels_path):
#                for label_file in os.listdir(labels_path):
#                    label_file_path = os.path.join(labels_path, label_file)
#                    if os.path.isfile(label_file_path):
#                        with open(label_file_path, 'r') as file:
#                            content = file.read().strip()
#                            if content:  # Check if the file content is not empty
#                                labels_con_peces.append(label_file_path)


if __name__ == "__main__":
    format = "yolov8-obb"
    workspaces = ["memristor", "memristor", "alejandro-guerrero-zihxm", "alejandro-guerrero-zihxm"]
    projects = ["deepfish-segmentation-t7gmr", "salmon-sxxri", "shiny_salmons", "shiny_salmons"]
    versions = [1, 1, 2, 4]
    names = ["Deepfish", "Salmones", "ShinySalmonsV2", "ShinySalmonsV4"]

    txt_con_peces = lista_txt_con_clase("datasets/roboflow/Deepfish")
    img_con_peces = lista_img_con_clase(txt_con_peces)
    for xd, xp in zip(txt_con_peces, img_con_peces):
        print(xd)
        print(xp)
        print()

    # Crear todos los datasets necesarios
    #for workspace, project, version, name in zip(workspaces, projects, versions, names):
    #    dataset_path = os.path.join("datasets/roboflow", name)
    #    dataset = download_roboflow_dataset(workspace, project, version, format, dataset_path)
