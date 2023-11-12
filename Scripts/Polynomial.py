#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:22:20 2023

@author: Equipo DSA
"""

# Importe el conjunto de datos de diabetes y divídalo en entrenamiento y prueba usando scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

db = load_diabetes()
X = db.data
y = db.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Importar Librerias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder


#Importe MLFlow para registrar los experimentos, el regresor de bosques aleatorios y la métrica de error cuadrático medio
import mlflow
import mlflow.sklearn
import datetime
from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error

# defina el servidor para llevar el registro de modelos y artefactos
#mlflow.set_tracking_uri('http://localhost:5000')


# registre el experimento
experiment = mlflow.set_experiment("Booster")
#nombre incremental
time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#nombre incremental

run_name=f"Booster-{time}"
# Aquí se ejecuta MLflow sin especificar un nombre o id del experimento. MLflow los crea un experimento para este cuaderno por defecto y guarda las características del experimento y las métricas definidas. 
# Para ver el resultado de las corridas haga click en Experimentos en el menú izquierdo. 
with mlflow.start_run(run_name=run_name):
    # defina los parámetros del modelo
    n_estimators = 200 
    max_depth = 6
    max_features = 4
    # Cree el modelo con los parámetros definidos y entrénelo
    rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
    rf.fit(X_train, y_train)
    # Realice predicciones de prueba
    predictions = rf.predict(X_test)
  
    # Registre los parámetros
    mlflow.log_param("num_trees", n_estimators)
    mlflow.log_param("maxdepth", max_depth)
    mlflow.log_param("max_feat", max_features)
  
    # Registre el modelo
    mlflow.sklearn.log_model(rf, "random-forest-model")
  
    # Cree y registre la métrica de interés
    mse = mean_squared_error(y_test, predictions)
    mlflow.log_metric("mse", mse)
    print(mse)



with open('../data/Evaluaciones_Agropecuarias_Municipales_EVA.csv', 'r', encoding='utf-8') as file:
    evaluaciones = pd.read_csv(file)

columnas = ['NOMBRE_CULTIVO', 'ANIO', 'NOMBRE_MUNICIPIO', 'RENDIMIENTO_TONELADAS_HA']
df = evaluaciones[columnas]

# Eliminar filas con valores nulos en la columna 'RENDIMIENTO_TONELADAS_HA'
df.dropna(subset=['RENDIMIENTO_TONELADAS_HA'], inplace=True)

# Codificar variables categóricas
le = LabelEncoder()
df['NOMBRE_CULTIVO'] = le.fit_transform(df['NOMBRE_CULTIVO'])
df['NOMBRE_MUNICIPIO'] = le.fit_transform(df['NOMBRE_MUNICIPIO'])

# Definir variables X y Y
X = ['NOMBRE_CULTIVO', 'ANIO', 'NOMBRE_MUNICIPIO']
Y = 'RENDIMIENTO_TONELADAS_HA'
X = df[X]
Y = df[Y]

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, Y)

# Crear características polinómicas
degree = 2
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)

# Crear y entrenar el modelo
modelo = LinearRegression()
modelo.fit(X_train_poly, y_train)

# Predecir con el conjunto de prueba
y_pred = modelo.predict(X_test_poly)

# Calcular métricas de evaluación
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Imprimir resultados
print(f'MSE: {mse}')
print(f'MAE: {mae}')