#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 09:44:21 2024

@author: tori
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import NearMiss

excel_file = "./Dataset_Year_2020.xlsx"
data = pd.read_excel(excel_file)

# print("Información general de los datos (Antes):")
# print(data.info())

# Convertir las columnas categóricas en valores numéricos (1 a N)
data['Type of clients'] = pd.factorize(data['Type of clients'])[0] + 1
data['Type of installation'] = pd.factorize(data['Type of installation'])[0] + 1


# Información general después de la conversión
print("Información general de los datos (Después):")
print(data.info())

# Estadísticas descriptivas de las variables numéricas
print("\nEstadísticas descriptivas de las variables numéricas:")
print(data.describe())

# Definir características (X) y variable objetivo (y)
X = data.drop('Burned transformers 2020', axis=1)
y = data['Burned transformers 2020']

# Aplicar NearMiss para el undersampling
undersample = NearMiss(version=1)
X_resampled, y_resampled = undersample.fit_resample(X, y)

# Combinar los datos resampleados en un DataFrame
data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['Burned transformers 2020'])], axis=1)

# Estandarización de las columnas numéricas
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

for column in numeric_columns:
    mean = data[column].mean()
    std = data[column].std()
    data[column] = (data[column] - mean) / std

print("Datos estandarizados:")
print(data.describe())

# Nuevo conteo
print(data['Burned transformers 2020'].value_counts())

# Crear histogramas
data.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()

# Crear la matriz de correlación
corr_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title('Matriz de correlación')
plt.show()