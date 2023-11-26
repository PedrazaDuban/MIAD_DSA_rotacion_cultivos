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

with open('data/cultivos.csv', 'r', encoding='utf-8') as file:
    dataset = pd.read_csv(file)
Y=dataset['RENDIMIENTO_TONELADAS_HA']
X=dataset[['ANIO','NOMBRE_CULTIVO', 'NUM_CLUSTERS']]
X = X.replace(',','', regex=True)
# Create arrary of categorial variables to be encoded
categorical_cols = ['CULTIVO']
le = LabelEncoder()
# apply label encoder on categorical feature columns
X[categorical_cols] = X[categorical_cols].apply(lambda col: le.fit_transform(col))
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=13
)



import datetime
from sklearn.preprocessing import robust_scale
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets, ensemble
from sklearn.metrics import mean_squared_error

n_estimators = 500 
max_depth = 4
min_samples_split = 5
learning_rate = 0.01
loss = 'squared_error'
# Cree el modelo con los parámetros definidos y entrénelo
reg = ensemble.GradientBoostingRegressor(n_estimators = n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, learning_rate=learning_rate,loss=loss)
reg.fit(X_train, y_train)
