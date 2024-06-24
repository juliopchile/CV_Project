# Datasets
- Deepfish: disponible con descarga directa desde Roboflow
- Salmon: disponible con descarga desde One-drive https://usmcl-my.sharepoint.com/:f:/g/personal/julio_lopezb_sansano_usm_cl/EhFfwMzPsBVAtpc5rhund-QBJO7Cbiao084XnxQHPRUbpg?e=QmbrAB
- ShinySalmon: disponible con descarda desde Roboflow con invitación previa

# Código
- config: módulo con definición de paths y datasets.
- supersecrets: módulo neceario para la descarga por Roboflow, aquí se guarda la API_KEY.
- model_utils: funciones para descargar, cargar y exportar modelos.
- dataset_utils: funciones para descargar datasets desde roboflow y configurarlos con el formato de segmentación correcto.
- training: código para definir los hiperparámetros de entrenamiento y realizar el entrenamiento. Configurar según se necesite para cada experimento.
- main: código para testear los modelos (hacer inferencia con un set de imagenes de prueba).

# Directorios
