# Resumen
Este es el código utilizado para una tarea de la asignatura de *Visión por Computador*, impartida por el profesor Marcos Zúñiga durante el primer semestre de 2024 en la Universidad Técnica Federico Santa María. La tarea se divide en tres partes: parte 0 (SOTA), donde se estudia el estado del arte de alguna tarea de visión por computador; parte 1 (benchmark), donde busca replicar los resultados de algun trabajo o paper previo; y la parte 2 (algoritmo), donde se busca realizar una mejora, cambio o innovación al trabajo realizado en la parte 1. El código es par las partes 1 y 2.

- En la parte 1 se utiliza YOLO para realizar segmentación de instancias en peces, donde se replican resultados obtenidos por Alejandro Guerrero en su [memoria de titulación](https://repositorio.usm.cl/entities/tesis/e19ea8a6-b0eb-4727-903e-9c92f5d290bf) pero usando [YOLOv9](https://docs.ultralytics.com/models/yolov9/) en vez de [YOLOv8](https://docs.ultralytics.com/models/yolov8/).
- En la parte 2 se hace un intento de probar modelos [SAM](https://docs.ultralytics.com/models/sam/), [MobileSAM](https://docs.ultralytics.com/es/models/mobile-sam/) y [FastSAM](https://docs.ultralytics.com/es/models/fast-sam/) para segmentación de dos etapas, usando YOLOv9 de detección y luego algún modelo SAM para la segmentación, también se intenta hacer tracking con modelos SAM.

# Datasets
- Deepfish: disponible con descarga directa con [Roboflow](https://universe.roboflow.com/memristor/deepfish-segmentation-ocdlj).
- Salmon: disponible con descarga desde [One-Drive](https://usmcl-my.sharepoint.com/:f:/g/personal/julio_lopezb_sansano_usm_cl/EhFfwMzPsBVAtpc5rhund-QBJO7Cbiao084XnxQHPRUbpg?e=QmbrAB).
- ShinySalmon: disponible con descarga desde [Roboflow](https://app.roboflow.com/alejandro-guerrero-zihxm/shiny_salmons/4), con invitación previa.

**Todos los datasets están en el link de One-Drive.**

<table><thead>
  <tr>
    <th>Dataset</th>
    <th>Task</th>
    <th>Num. imgs</th>
    <th>Num. segm.</th>
    <th>Img. fondo</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="3">Deepfish<br>(620)</td>
    <td>Train</td>
    <td>309</td>
    <td>182</td>
    <td>161</td>
  </tr>
  <tr>
    <td>Validation</td>
    <td>125</td>
    <td>79</td>
    <td>59</td>
  </tr>
  <tr>
    <td>Testing</td>
    <td>186</td>
    <td>113</td>
    <td>90</td>
  </tr>
  <tr>
    <td rowspan="3">Salmon<br>(801)</td>
    <td>Train</td>
    <td>715</td>
    <td>3048</td>
    <td>322</td>
  </tr>
  <tr>
    <td>Validation</td>
    <td>86</td>
    <td>496</td>
    <td>34</td>
  </tr>
  <tr>
    <td>Testing</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
  </tr>
  <tr>
    <td rowspan="3">ShinySalmonsV4<br>(130)</td>
    <td>Train</td>
    <td>124</td>
    <td>1189</td>
    <td>0</td>
  </tr>
  <tr>
    <td>Validation</td>
    <td>4</td>
    <td>30</td>
    <td>0</td>
  </tr>
  <tr>
    <td>Testing</td>
    <td>2</td>
    <td>16</td>
    <td>0</td>
  </tr>
</tbody></table>

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
- CustomFastSAMPrompt: clase modificada para realizar prompting con FastSAM, necesaria para múltiples instancias por imagen.

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
- [Parte 0](https://usmcl-my.sharepoint.com/:b:/g/personal/julio_lopezb_sansano_usm_cl/EXbcZc_b9RdHsuEVIhi4WdsBqIGwysmvGtxDmGA0u7RgCQ?e=G0JfYQ) Pre-informe con estado del arte (bien malo, no lo recomiendo).
- [Parte 1](https://usmcl-my.sharepoint.com/:b:/g/personal/julio_lopezb_sansano_usm_cl/Ec5BoCSXgzZGqsnf7QvZ_OYBnIP-aIplpm2Kg1NTtxQgCg?e=d7HRgF) Entrenamiento de Yolov9-seg y comparación con el estado del arte.
- [Parte 2](https://usmcl-my.sharepoint.com/:b:/g/personal/julio_lopezb_sansano_usm_cl/EZMhsb5AmF1HlquAtW8LXK0B8Q_kq_ZVq0RjWjpXAYWBkw?e=doxJHZ) Segmentación con SAM, cuantización TensorRT y tracking.

# Resultados en segmentación

<table>
  <caption>
    <b>Métricas de validación a los mejores modelos YOLO-seg entrenados, para cada dataset.</b>
  </caption>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Modelo</th>
      <th>Optimizador</th>
      <th>Formato</th>
      <th>F1 Score</th>
      <th>mAP50</th>
      <th>mAP50-95</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="8">Deepfish</td>
      <td rowspan="2">Yolov8x-seg</td>
      <td rowspan="2">SGD</td>
      <td>Float32</td>
      <td>0.884</td>
      <td>0.938</td>
      <td>0.728</td>
    </tr>
    <tr>
      <td>Int8</td>
      <td>0.871</td>
      <td>0.921</td>
      <td>0.748</td>
    </tr>
    <tr>
      <td rowspan="2">Yolov8l-seg + DUP L.O.</td>
      <td rowspan="2">SGD</td>
      <td>Float32</td>
      <td>0.924</td>
      <td>0.928</td>
      <td>0.730</td>
    </tr>
    <tr>
      <td>Int8</td>
      <td>0.703</td>
      <td>0.656</td>
      <td>0.487</td>
    </tr>
    <tr>
      <td rowspan="2">Yolov9c-seg F.T.</td>
      <td rowspan="2">SGD</td>
      <td>Float32</td>
      <td>0.987</td>
      <td>0.994</td>
      <td>0.823</td>
    </tr>
    <tr>
      <td>Int8</td>
      <td>0.5321</td>
      <td>0.457</td>
      <td>0.291</td>
    </tr>
    <tr>
      <td rowspan="2">Yolov9c-seg</td>
      <td rowspan="2">AdamW</td>
      <td>Float32</td>
      <td>0.980</td>
      <td>0.990</td>
      <td>0.821</td>
    </tr>
    <tr>
      <td>Int8</td>
      <td>0.328</td>
      <td>0.308</td>
      <td>0.202</td>
    </tr>
    <tr>
      <td rowspan="8">Salmon</td>
      <td rowspan="2">Yolov8l-seg.</td>
      <td rowspan="2">SGD</td>
      <td>Float32</td>
      <td>0.678</td>
      <td>0.709</td>
      <td>0.406</td>
    </tr>
    <tr>
      <td>Int8</td>
      <td>0.5466</td>
      <td>0.523</td>
      <td>0.287</td>
    </tr>
    <tr>
      <td rowspan="2">Yolov8l-seg L.O.</td>
      <td rowspan="2">Adam</td>
      <td>Float32</td>
      <td>0.632</td>
      <td>0.656</td>
      <td>0.367</td>
    </tr>
    <tr>
      <td>Int8</td>
      <td>0.479</td>
      <td>0.448</td>
      <td>0.225</td>
    </tr>
    <tr>
      <td rowspan="4">Yolov9c-seg F.T.</td>
      <td rowspan="2">SGD</td>
      <td>Float32</td>
      <td>0.675</td>
      <td>0.688</td>
      <td>0.400</td>
    </tr>
    <tr>
      <td>Int8</td>
      <td>0.306</td>
      <td>0.255</td>
      <td>0.107</td>
    </tr>
    <tr>
      <td rowspan="2">AdamW</td>
      <td>Float32</td>
      <td>0.672</td>
      <td>0.674</td>
      <td>0.358</td>
    </tr>
    <tr>
      <td>Int8</td>
      <td>0.494</td>
      <td>0.439</td>
      <td>0.202</td>
    </tr>
    <tr>
      <td rowspan="4">ShinySalmonsV4</td>
      <td rowspan="2">Yolov8l-seg Retrained<br />from Salmon</td>
      <td rowspan="2">SGD</td>
      <td>Float32</td>
      <td>0.592</td>
      <td>0.635</td>
      <td>0.513</td>
    </tr>
    <tr>
      <td>Int8</td>
      <td>0.711</td>
      <td>0.783</td>
      <td>0.606</td>
    </tr>
    <tr>
      <td rowspan="2">Yolov9c-seg Retrained<br />from Salmon</td>
      <td rowspan="2">SGD</td>
      <td>Float32</td>
      <td>0.833</td>
      <td>0.876</td>
      <td>0.696</td>
    </tr>
    <tr>
      <td>Int8</td>
      <td>0.699</td>
      <td>0.785</td>
      <td>0.430</td>
    </tr>
  </tbody>
</table>

