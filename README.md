# Datasets
- Deepfish: disponible con descarga directa desde Roboflow
- Salmon: disponible con descarga desde [One-Drive](https://usmcl-my.sharepoint.com/:f:/g/personal/julio_lopezb_sansano_usm_cl/EhFfwMzPsBVAtpc5rhund-QBJO7Cbiao084XnxQHPRUbpg?e=QmbrAB)
- ShinySalmon: disponible con descarga desde Roboflow con invitación previa
- De todas formas todos los datasets están en el link de One-Drive.

# Código
- config: módulo con definición de paths y datasets.
- supersecrets: módulo necesario para la descarga por Roboflow, aquí se guarda la API_KEY de roboflow. (no está en github)
- model_utils: funciones para descargar, cargar y exportar modelos.
- dataset_utils: funciones para descargar datasets desde roboflow y configurarlos con el formato de segmentación correcto.
- training: código para definir los hiperparámetros de entrenamiento y realizar el entrenamiento.
- training_runs: código para definir los experimentos y entrenar.
- main: main hub desde donde poder llevar a cabo los entrenamientos definidos en training_runs.
- validation: código para realizar validación de modelos de forma ordenada.
- use_sam: código con funciones para utilizar SAM en conjunto con un modelo de detección.
- CustomFastSAMPrompt: clase custom para realizar prompting con FastSAM, necesaria para múltiples instancias por imagen.

# Directorios
- dataset_yaml: contiene los archivos yaml para los datasets de segmentación guardados en coco_converted.
- coco_converted: aquí se encuentran los labels e imágenes del dataset de segmentación.
- yolo_dataset: aquí se encuentran los datasets de detección.
- test_files: imágenes de prueba.
- test_files_salmon: imágenes de prueba, solo del dataset salmon.
- test_videos: videos de prueba.
- models: donde se guardan los modelos y los resultados de entrenamiento. (creado por código/no incluido en github)
- runs: donde se guardan resultados de validación y testeo. (creado por código/no incluido en github)

# Videos e imágenes de inferencia
- Imágenes de resultados para los distintos modelos SAM en la carpeta [Segmentaciones](Segmentaciones)
- Videos probando inferencia en video en las listas de reproducción siguientes:
  - [SAM](https://www.youtube.com/playlist?list=PLaAjsJBsA0UTrqkqmjRvsd4QUNqs_Ygb_) (SAM_b - SAM_l - MobileSAM - FastSAM)
  - [Tracking](https://www.youtube.com/playlist?list=PLaAjsJBsA0UT4_vWxxlujuwxjat6lsZ-I) (Yolov8 - Yolov9 - FastSAM)
