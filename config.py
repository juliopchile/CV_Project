# ? ----MODELOS---- ? #

# Path donde guardar los modelos sin entrenar
backbones_directory = 'models/backbone'

# Listas de todos los modelos disponibles
yolo_models = ['yolov8n-seg', 'yolov8s-seg', 'yolov8m-seg', 'yolov8l-seg', 'yolov8x-seg', 'yolov9c-seg',
               'yolov9e-seg', 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x', 'yolov9c', 'yolov9e']
sam_models = ['sam_l', 'sam_b', 'mobile_sam']
fast_sam_models = ['FastSAM-s', 'FastSAM-x']
nas_models = ['yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l']

# Modelos descargables con código
downloadable_models = (yolo_models + sam_models + fast_sam_models)

# Modelos usables
loadable_models = downloadable_models + nas_models


# ? ----DATASETS----- ? #

# Path donde guardar todos los datasets
datasets_directory = "datasets"
coco_labels_directory = "coco_converted"

# Diccionario usado para descargar datasets
datasets_link = {
    "Deepfish": dict(workspace="memristor", project="deepfish-segmentation-ocdlj", version=3, name="Deepfish"),
    "Deepfish_LO": dict(workspace="memristor", project="deepfish-segmentation-ocdlj", version=4, name="Deepfish_LO"),
    "Salmon": dict(workspace="memristor", project="salmones-ji1wj", version=5, name="Salmones"),
    "Salmon_LO": dict(workspace="memristor", project="salmones-ji1wj", version=6, name="Salmones_LO"),
    "Shiny_v2": dict(workspace="alejandro-guerrero-zihxm", project="shiny_salmons", version=2, name="ShinySalmonsV2"),
    "Shiny_v4": dict(workspace="alejandro-guerrero-zihxm", project="shiny_salmons", version=4, name="ShinySalmonsV4"),
}


# Diccionario usado para acceder a los datasets para los experimentos. (Comentar/borrar los que no se quieran entrenar)
datasets_path = {
    "Salmones": "dataset_yaml_files/salmones.yaml",
    "Deepfish": "dataset_yaml_files/deepfish.yaml",
    #"ShinySalmonsV4": "dataset_yaml_files/shiny_salmons_v4.yaml"
}
