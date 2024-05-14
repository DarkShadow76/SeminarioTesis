#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 09:44:21 2024

@author: tori
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, r2_score

excel_file = "./Dataset_Year_2020.xlsx"
data = pd.read_excel(excel_file)

print("Información general de los datos(Antes):")
print(data.info())

# Obtén las categorías únicas de las columnas categóricas
categorias_cliente = data['Type of clients'].unique()
categorias_instalacion = data['Type of installation'].unique()

# Categorizar la variable 'Type of clients'
data['Type of clients'] = data['Type of clients'].astype('category').cat.codes

# Categorizar la variable 'Type of installation'
data['Type of installation'] = data['Type of installation'].astype('category').cat.codes

# Creamos una instancia del StandardScaler
scaler = StandardScaler()

# Seleccionamos solo las columnas numéricas para estandarizar
columnas_numericas = data.select_dtypes(include=['int64','int8', 'float64']).columns

# Estandarizamos las columnas seleccionadas
data[columnas_numericas] = scaler.fit_transform(data[columnas_numericas])


print("Información general de los datos(Despues):")
print(data.info())

print("\nEstadísticas descriptivas de las variables numéricas:")
print(data.describe())


corr_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title('Matriz de correlación')
plt.show()

# Seleccionar todas las columnas numéricas excepto 'Burned transformers 2020'
columnas_numericas = data.select_dtypes(include=['int64', 'int8', 'float64']).columns.drop('Burned transformers 2020')

# Crear el box plot
plt.figure(figsize=(12, 8))
sns.boxplot(data=data[columnas_numericas])
plt.title('Box plot de las columnas numéricas')
plt.xlabel('Columnas')
plt.ylabel('Valores')
plt.xticks(rotation=45)  # Rotar las etiquetas del eje x para una mejor visualización
plt.show()

# División de los datos en conjunto de entrenamiento y conjunto de prueba
X = data.drop('Burned transformers 2020', axis=1)
y = data['Burned transformers 2020']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instancia de Regresor de Vectores de Soporte (SVR)
svm_regressor = SVR(kernel='linear')  # Usa SVR en lugar de SVC

# Entrenamiento del modelo de regresión
svm_regressor.fit(X_train, y_train)

# Predicción en el conjunto de prueba
y_pred = svm_regressor.predict(X_test)

# Evaluación del modelo de regresión
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Error cuadrático medio (MSE):", mse)
print("Coeficiente de determinación (R^2):", r2)