# Path donde guardar los modelos sin entrenar
backbone_path = 'models/backbone'

# Listas de todos los modelos
sam_models = ['sam_l', 'sam_b', 'mobile_sam', 'FastSAM-s', 'FastSAM-x']
yolo_nas_models = ['yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l']
yolov8_segmentation_models = ['yolov8n-seg', 'yolov8s-seg', 'yolov8m-seg', 'yolov8l-seg', 'yolov8x-seg']
yolov9_segmentation_models = ['yolov9c-seg', 'yolov9e-seg', ]
yolov8_detection_models = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
yolov9_detection_models = ['yolov9c', 'yolov9e']
model_to_download = (sam_models + yolov8_segmentation_models + yolov9_segmentation_models +
                     yolov8_detection_models + yolov9_detection_models)

# Datasets
datasets = {
    "Deepfish": {"version": 1, "model_format": "yolov8-obb", "location": "datasets/roboflow/Deepfish"},
    "Salmones": {"version": 1, "model_format": "yolov8-obb", "location": "datasets/roboflow/Salmones"},
    "ShinySalmonsV2": {"version": 2, "model_format": "yolov8-obb", "location": "datasets/roboflow/ShinySalmonsV2"},
    "ShinySalmonsV4": {"version": 4, "model_format": "yolov8-obb", "location": "datasets/roboflow/ShinySalmonsV4"}
}
