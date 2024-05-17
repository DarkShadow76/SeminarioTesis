#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 09:44:21 2024

@author: tori
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
<<<<<<< HEAD
from sklearn.svm import SVR
=======
from sklearn.utils import resample
>>>>>>> 0c9befc (undersampling)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, r2_score

excel_file = "./Dataset_Year_2020.xlsx"
data = pd.read_excel(excel_file)

print("Información general de los datos (Antes):")
print(data.info())

## Tecnica OverSampling
# Sparar Clases mayoritarias y minoritarias
majority_class = data[data['Burned transformers 2020'] == 0]
minority_class = data[data['Burned transformers 2020'] == 1]

# Oversample en la clase minoritaria
majority_class_undersampled = resample(majority_class,
                                       replace=False,  # sample without replacement
                                       n_samples=len(minority_class),  # match minority class size
                                       random_state=42)  # for reproducibility

# Cambinear ambas clases
data = pd.concat([majority_class_undersampled, minority_class])

# Nuevo conteo
print(data['Burned transformers 2020'].value_counts())

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
columnas_std = data.columns

# Estandarizamos las columnas seleccionadas
normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# data.columns = scaler.fit_transform(data[columnas_std])

data = normalized_data

print("Información general de los datos(Despues):")
print(data.info())

print("\nEstadísticas descriptivas de las variables numéricas:")
print(data.describe())


# Matriz
corr_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title('Matriz de correlación')
plt.show()

# Seleccionar todas las columnas numéricas excepto 'Burned transformers 2020'
columnas_numericas = data.select_dtypes(include=['int64', 'float64']).columns.drop('Burned transformers 2020')

# Crear el box plot

plt.figure(figsize=(20, 16))
sns.boxplot(data=data[columnas_numericas])
plt.title('Box plot de las variables')
plt.xlabel('Columnas')
plt.ylabel('Valores')
<<<<<<< HEAD
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
=======
plt.xticks(rotation=45)
plt.show()

print(data.head())
>>>>>>> 0c9befc (undersampling)
