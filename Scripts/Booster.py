#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11  2023

@author: Equipo DSA
"""

# Importe el conjunto de datos de diabetes y divídalo en entrenamiento y prueba usando scikit-learn
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.preprocessing import robust_scale
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

with open('data/Evaluaciones_Agropecuarias_Municipales_EVA.csv', 'r', encoding='utf-8') as file:
    evaluaciones = pd.read_csv(file)
evaluaciones2=evaluaciones[['CÓD. MUN.', 'CULTIVO']]
pivot = np.round(pd.pivot_table(evaluaciones2, index='CÓD. MUN.',
                                columns='CULTIVO', aggfunc= len, fill_value=0))
pivot.reset_index(inplace=True)
pivot['DPTOMPIO']=pivot['CÓD. MUN.']
municipios=gpd.read_file("data/MunicipiosVeredas19MB.json")
municipios['DPTOMPIO']=municipios[['DPTOMPIO']].apply(pd.to_numeric)
municipios2=municipios[['DPTOMPIO','geometry']]
pivot = pivot.astype({'CÓD. MUN.':'int'})
municipios3=pd.merge(municipios2,pivot, left_on='DPTOMPIO', right_on='DPTOMPIO')
municipios3 = municipios3.iloc[: , 1:]
db_scaled = robust_scale(municipios3.loc[:, ~municipios3.columns.isin(['geometry', 'CÓD. MUN.'])])
kmeans = KMeans(n_clusters=10)
# Set the seed for reproducibility
np.random.seed(1234)
# Run K-Means algorithm
k10cls = kmeans.fit(db_scaled)
municipios3["k10cls"] = k10cls.labels_
evaluaciones3 = pd.merge(evaluaciones,municipios3[['CÓD. MUN.','k10cls']],on='CÓD. MUN.', how='left')
evaluaciones3=evaluaciones3.dropna(subset='Rendimiento\n(t/ha)')
evaluaciones3['Rendimiento\n(t/ha)'] = evaluaciones3[evaluaciones3['Rendimiento\n(t/ha)'] < 25]['Rendimiento\n(t/ha)']
evaluaciones3=evaluaciones3.dropna(subset='Rendimiento\n(t/ha)')
evaluaciones3=evaluaciones3.dropna(subset='AÑO')
evaluaciones3=evaluaciones3.dropna(subset='CULTIVO')
evaluaciones3=evaluaciones3.dropna(subset='k10cls')
Y=evaluaciones3['Rendimiento\n(t/ha)']
X=evaluaciones3[['AÑO','CULTIVO','k10cls']]
X = X.replace(',','', regex=True)
# Create arrary of categorial variables to be encoded
categorical_cols = ['CULTIVO']
le = LabelEncoder()
# apply label encoder on categorical feature columns
X[categorical_cols] = X[categorical_cols].apply(lambda col: le.fit_transform(col))
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=13
)


#Importe MLFlow para registrar los experimentos, el regresor de bosques aleatorios y la métrica de error cuadrático medio
import mlflow
import mlflow.sklearn
import datetime
from sklearn.preprocessing import robust_scale
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets, ensemble
from sklearn.metrics import mean_squared_error

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
    max_depth = 4
    min_samples_split = 5
    learning_rate = 0.01
    loss = 'squared_error'
    # Cree el modelo con los parámetros definidos y entrénelo
    reg = ensemble.GradientBoostingRegressor(n_estimators = n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, learning_rate=learning_rate,loss=loss)
    reg.fit(X_train, y_train)
    # Realice predicciones de prueba
    predictions = reg.predict(X_test)
  
    # Registre los parámetros
    mlflow.log_param("num_estimators", n_estimators)
    mlflow.log_param("maxdepth", max_depth)
    mlflow.log_param("min_samples_split", min_samples_split)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("loss", loss)
  
    # Registre el modelo
    mlflow.sklearn.log_model(reg, "gradien-boosting-regressor")
  
    # Cree y registre la métrica de interés
    mse = mean_squared_error(y_test, predictions)
    mlflow.log_metric("mse", mse)
    print(mse)
