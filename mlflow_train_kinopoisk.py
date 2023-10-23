import os

# Установка переменных окружения в Unix-подобных системах (Mac, Linux)
os.system('export MLFLOW_TRACKING_URI=http://localhost:5000')
os.system('export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000')

# Установка переменных окружения в Windows
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'

os.system('mlflow run . --experiment-name=kinopoisk')