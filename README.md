# Datasets
- Deepfish: disponible con descarga directa desde [Roboflow](https://universe.roboflow.com/memristor/deepfish-segmentation-ocdlj)
- Salmon: disponible con descarga desde [One-Drive](https://usmcl-my.sharepoint.com/:f:/g/personal/julio_lopezb_sansano_usm_cl/EhFfwMzPsBVAtpc5rhund-QBJO7Cbiao084XnxQHPRUbpg?e=QmbrAB)
- ShinySalmon: disponible con descarga desde Roboflow con invitación previa.

**Todos los datasets están en el link de One-Drive.**

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

# Informes en formato paper
- [Parte 1](https://usmcl-my.sharepoint.com/:b:/g/personal/julio_lopezb_sansano_usm_cl/Ec5BoCSXgzZGqsnf7QvZ_OYBnIP-aIplpm2Kg1NTtxQgCg?e=d7HRgF) Entrenamiento de Yolov9-seg y comparación con el estado del arte.
- [Parte 2](https://usmcl-my.sharepoint.com/:b:/g/personal/julio_lopezb_sansano_usm_cl/EZMhsb5AmF1HlquAtW8LXK0B8Q_kq_ZVq0RjWjpXAYWBkw?e=doxJHZ) Segmentación con SAM, cuantización y tracking.

# Resultados
| Dataset           | Modelo                                     | Opt.  | Type    | F1 Score | mAP50 | mAP50‑95 |
|-------------------|--------------------------------------------|-------|---------|----------|-------|----------|
| **Deepfish**      | Yolov8x-seg                                | SGD   | Float32 | 0.884    | 0.938 | 0.728    |
|                   | Yolov8x-seg                                | SGD   | Int8    | 0.871    | 0.921 | 0.748    |
|                   | Yolov8l-seg + DUP L.O.                     | SGD   | Float32 | 0.924    | 0.928 | 0.730    |
|                   | Yolov8l-seg + DUP L.O.                     | SGD   | Int8    | 0.703    | 0.656 | 0.487    |
|                   | Yolov9c-seg F.T.                           | SGD   | Float32 | 0.987    | 0.994 | 0.823    |
|                   | Yolov9c-seg F.T.                           | SGD   | Int8    | 0.5321   | 0.457 | 0.291    |
|                   | Yolov9c-seg                                | AdamW | Float32 | 0.980    | 0.990 | 0.821    |
|                   | Yolov9c-seg                                | AdamW | Int8    | 0.328    | 0.308 | 0.202    |
| **Salmon**        | Yolov8l-seg.                               | SGD   | Float32 | 0.678    | 0.709 | 0.406    |
|                   | Yolov8l-seg.                               | SGD   | Int8    | 0.5466   | 0.523 | 0.287    |
|                   | Yolov8l-seg L.O.                           | Adam  | Float32 | 0.632    | 0.656 | 0.367    |
|                   | Yolov8l-seg L.O.                           | Adam  | Int8    | 0.479    | 0.448 | 0.225    |
|                   | Yolov9c-seg F.T.                           | SGD   | Float32 | 0.675    | 0.688 | 0.400    |
|                   | Yolov9c-seg F.T.                           | SGD   | Int8    | 0.306    | 0.255 | 0.107    |
|                   | Yolov9c-seg F.T.                           | AdamW | Float32 | 0.672    | 0.674 | 0.358    |
|                   | Yolov9c-seg F.T.                           | AdamW | Int8    | 0.494    | 0.439 | 0.202    |
| **ShinySalmonsV4**| Yolov8l-seg Retrained from Salmon          | SGD   | Float32 | 0.592    | 0.635 | 0.513    |
|                   | Yolov8l-seg Retrained from Salmon          | SGD   | Int8    | 0.711    | 0.783 | 0.606    |
|                   | Yolov9c-seg Retrained from Salmon          | SGD   | Float32 | 0.833    | 0.876 | 0.696    |
|                   | Yolov9c-seg Retrained from Salmon          | SGD   | Int8    | 0.699    | 0.785 | 0.430    |
