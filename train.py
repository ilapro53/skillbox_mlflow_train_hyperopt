import boto3
import pandas as pd
import numpy as np
from io import StringIO
import os

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.sklearn

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK



# Распаковка словаря
def f_unpack_dict(dct):
    res = {}
    for (k, v) in dct.items():
        if isinstance(v, dict):
            res = {**res, **f_unpack_dict(v)}
        else:
            res[k] = v
            
    return res



# Инициализация клиента
print('Инициализация клиента...')
s3 = boto3.client('s3',
                  endpoint_url='http://localhost:9000',
                  aws_access_key_id='minio',
                  aws_secret_access_key='minio123')



# Считывание данных
print('Считывание данных...')
obj = s3.get_object(Bucket='datasets', Key='kinopoisk_train.csv')
data = obj['Body'].read().decode('utf-8')
df = pd.read_csv(StringIO(data))



# Установка переменных окружения в Unix-подобных системах (Mac, Linux)
os.system('export MLFLOW_TRACKING_URI=http://localhost:5000')
os.system('export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000')

# Установка переменных окружения в Windows
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'



print('Подготовка к обучению...')
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2)



# Алгоритмы (пайплайны)
models = dict() # Задаем вручную

models['text_clf_v_mnnb'] = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('mnnb', SGDClassifier()),
])



# Создание функций для оптимизации
objectives = dict() # Создается автоматически как в models

for model_name, model in models.items():
    def objective(params):
        params = f_unpack_dict(params)
        print(params)

        model = models[model_name]
        model.set_params(**params)
        model.fit(X_train, y_train)
        print(model.get_params())

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {'loss': -accuracy, 'params': params, 'status': STATUS_OK}
    
    objectives[model_name] = objective

del(objective)



# Оптимизация параметров
print('Обучение (оптимизация параметров)...')
spaces = dict() # Задаем вручную

spaces['text_clf_v_logreg'] = {
        'group_by__logreg__penalty': hp.choice('hyper_param_groups',
            [
                {
                    'logreg__penalty': hp.choice('penalty_block1', ['l2']),
                    'logreg__solver': hp.choice('solver_block1', ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']),
                },

                {
                    'logreg__penalty': hp.choice('penalty_block3', ['l1']),
                    'logreg__solver': hp.choice('solver_block3', ['liblinear', 'saga']),
                },
                {
                    'logreg__penalty': hp.choice('penalty_block4', [None]),
                    'logreg__solver': hp.choice('solver_block4', ['lbfgs', 'newton-cg', 'sag', 'saga']),
                },
            ]),
        'logreg__class_weight': hp.choice('class_weight', ['balanced', None]),
    }

all_trails = dict() # Создается автоматически как в spaces

for model_name, space in spaces.items():
    print(f'Поиск параметров для {model_name}...')
    all_trails[model_name] = Trials()
    best = fmin(
        fn=objectives[model_name], 
        space=spaces[model_name], 
        algo=tpe.suggest, 
        max_evals=40,
        trials=all_trails[model_name]
    )



print('Получение лучшего результата')
# Получение лучшего результата
best_model_name = min(all_trails.items(), key=lambda x: float(x[1].best_trial['result']['loss']))[0]
best_model_trails = all_trails[best_model_name]
best_description = 'Pipeline(steps=' + str([type(s[1]).__name__ for s in models[best_model_name].steps]) + ')'

print('Лучшая модель:', best_model_name)
print('Точность лучшей модели:', -best_model_trails.best_trial['result']['loss'])



# Создание баккита "mlflow"
try:
    s3.create_bucket(Bucket='mlflow')
    print('Баккит "mlflow" создан')
except s3.exceptions.BucketAlreadyOwnedByYou:
    print('Баккит "mlflow" уже существует')



# Настройка клиента boto3
print('Настройка клиента boto3...')
boto3.setup_default_session(
    aws_access_key_id='minio',
    aws_secret_access_key='minio123',
    region_name='us-west-1'  # или другой регион, если это применимо
)


# Логирование в MLflow...
print('Логирование в MLflow...')
description = 'Pipeline(steps=' + str([type(s[1]).__name__ for s in models[best_model_name].steps]) + ')'
with mlflow.start_run() as run:
    # Логирование параметров и метрик
    mlflow.log_param("model_type", description)
    mlflow.log_metric("accuracy", -best_model_trails.best_trial['result']['loss'])
    
    # Логирование модели
    mlflow.sklearn.log_model(models[best_model_name], "model", registered_model_name="MyOptimizedModel")

print('Готово!')

# import os
# print(os.system('pip freeze'))