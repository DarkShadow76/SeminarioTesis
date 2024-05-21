#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 09:44:21 2024

@author: tori
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import InstanceHardnessThreshold

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import BorderlineSMOTE

# Leer el archivo Excel
excel_file = "./Dataset_Year_2020.xlsx"
data = pd.read_excel(excel_file)

# Imprimir información general antes del procesamiento
print("Información general de los datos (Antes):")
print(data.info())

# Comprobar las categorías únicas antes de la conversión
print("Categorías únicas antes de la conversión:")
print(data['Type of clients'].unique())
print(data['Type of installation'].unique())

# Convertir las columnas categóricas en valores numéricos (1 a N)
data['Type of clients'] = pd.factorize(data['Type of clients'])[0] + 1
data['Type of installation'] = pd.factorize(data['Type of installation'])[0] + 1

# Comprobar las categorías después de la conversión
print("Categorías después de la conversión:")
print(data['Type of clients'].unique())
print(data['Type of installation'].unique())

# Aplicar estandarización según la fórmula Xestandarizado = (X - μ) / σ
data_standardized = data.copy()
for column in data.columns:
    if column != 'Burned transformers 2020':
        mu = data[column].mean()
        sigma = data[column].std()
        data_standardized[column] = (data[column] - mu) / sigma

# Definir las características y la variable objetivo
X = data_standardized.drop(columns=['Burned transformers 2020'])
y = data_standardized['Burned transformers 2020'].astype(int)

# Aplicar undersampling utilizando el método Tomek Links
# Probar con CondensedNearestNeighbour
undersampler = InstanceHardnessThreshold()
X_resampled, y_resampled = undersampler.fit_resample(X, y)

# Combinar las características y la variable objetivo nuevamente
data_resampled = pd.DataFrame(X_resampled, columns=X.columns)
data_resampled['Burned transformers 2020'] = y_resampled

# Imprimir la nueva distribución de la variable objetivo
print("Distribución de 'Burned transformers 2020' después del undersampling:")
print(data_resampled['Burned transformers 2020'].value_counts())


"""
# Generar histogramas
data_resampled.hist(bins=30, figsize=(15, 10))
plt.suptitle('Histogramas de las características después del preprocesamiento')
plt.show()

# Generar matriz de correlación
corr_matrix = data_resampled.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title('Matriz de correlación')
plt.show()

# Generar box plots
data_resampled.drop(columns=['Burned transformers 2020']).plot(kind='box', figsize=(15, 10))
plt.title('Box plot de las características después del preprocesamiento')
plt.xticks(rotation=90)
plt.show()
"""