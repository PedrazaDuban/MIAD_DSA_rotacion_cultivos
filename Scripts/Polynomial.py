#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:22:20 2023

@author: Equipo DSA
"""

# Importar Librerias
import pandas as pd
import numpy as np
import mlflow.sklearn
import mlflow
import geopandas as gpd
import datetime
from sklearn.preprocessing import robust_scale
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, robust_scale
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn import metrics

# defina el servidor para llevar el registro de modelos y artefactos
#mlflow.set_tracking_uri('http://localhost:5000')

evaluaciones = pd.read_csv('/workspaces/MIAD_DSA_rotacion_cultivos/data/Evaluaciones_Agropecuarias_Municipales_EVA.csv')

municipios = gpd.read_file('/workspaces/MIAD_DSA_rotacion_cultivos/data/MunicipiosVeredas19MB.json')

# Filtrar columnas de interés en el DataFrame de evaluaciones
evaluaciones2 = evaluaciones[['CÓD. MUN.', 'CULTIVO']]

# Crear una tabla pivot
pivot = np.round(pd.pivot_table(evaluaciones2, index='CÓD. MUN.',
                                columns='CULTIVO', aggfunc=len, fill_value=0))
pivot.reset_index(inplace=True)
pivot['DPTOMPIO'] = pivot['CÓD. MUN.']

# Asumiendo que 'municipios' es un DataFrame existente
municipios['DPTOMPIO'] = municipios[['DPTOMPIO']].apply(pd.to_numeric)
municipios2 = municipios[['DPTOMPIO', 'geometry']]
pivot = pivot.astype({'CÓD. MUN.': 'int'})
municipios3 = pd.merge(municipios2, pivot, left_on='DPTOMPIO', right_on='DPTOMPIO')
municipios3 = municipios3.iloc[:, 1:]

# Calcular distancias pairwise
distances = metrics.pairwise_distances(
    municipios3.loc[:, ~municipios3.columns.isin(['geometry', 'CÓD. MUN.'])].head()
).round(4)

# Escalar los datos
db_scaled = robust_scale(municipios3.loc[:, ~municipios3.columns.isin(['geometry', 'CÓD. MUN.'])])

# Aplicar K-Means
kmeans = KMeans(n_clusters=10)
np.random.seed(1234)
k10cls = kmeans.fit(db_scaled)
k10cls.labels_[:5]

# Agregar las etiquetas al DataFrame de municipios
municipios3["k10cls"] = k10cls.labels_

# Fusionar DataFrame de evaluaciones y municipios3
evaluaciones3 = pd.merge(evaluaciones, municipios3[['CÓD. MUN.', 'k10cls']], on='CÓD. MUN.', how='left')

# Filtrar y limpiar datos
evaluaciones3 = evaluaciones3.dropna(subset='Rendimiento\n(t/ha)')
evaluaciones3 = evaluaciones3.dropna(subset='AÑO')
evaluaciones3 = evaluaciones3.dropna(subset='CULTIVO')
evaluaciones3 = evaluaciones3.dropna(subset='k10cls')

# Seleccionar columnas de interés
columnas = ['CULTIVO', 'AÑO', 'k10cls', 'Rendimiento\n(t/ha)']
df = evaluaciones3[columnas]

# Eliminar filas con valores nulos en la columna 'Rendimiento\n(t/ha)'
df.dropna(subset=['Rendimiento\n(t/ha)'], inplace=True)

# Codificar variables categóricas
le = LabelEncoder()
df['CULTIVO'] = le.fit_transform(df['CULTIVO'])
# df['MUNICIPIO'] = le.fit_transform(df['MUNICIPIO'])

# Definir variables X y Y
X = ['CULTIVO', 'AÑO', 'k10cls']
Y = 'Rendimiento\n(t/ha)'
X = df[X]
Y = df[Y]

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# Registrar el experimento
experiment = mlflow.set_experiment("Polynomial")
# Nombre incremental
time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

run_name = f"Polynomial-{time}"

with mlflow.start_run(run_name=run_name):
    # Define Parametros
    # Crear características polinómicas
    degree = 2    
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.fit_transform(X_test)

    # Crear y entrenar el modelo de regresión lineal
    modelo = LinearRegression()
    modelo.fit(X_train_poly, y_train)

    # Predecir con el conjunto de prueba
    y_pred = modelo.predict(X_test_poly)

    # Registrar los Parámetros
    mlflow.log_param("Degree_Polynomial", degree)
    mlflow.log_param("clusters", 10)

    # Registrar el Modélo
    mlflow.sklearn.log_model(modelo, "Polynomial")

    # Calcular métricas de evaluación
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    
    # Imprimir resultados
    print(f'MSE: {mse}')
    print(f'MAE: {mae}')

    # Finaliza el tracking de experimentos con MLflow
    mlflow.end_run()