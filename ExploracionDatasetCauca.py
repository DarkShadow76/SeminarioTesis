#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 09:44:21 2024

@author: tori
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

excel_file = "./Dataset_Year_2020.xlsx"
data = pd.read_excel(excel_file)

print("Información general de los datos:")
print(data.info())

print("\nEstadísticas descriptivas de las variables numéricas:")
print(data.describe())

data.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()

corr_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title('Matriz de correlación')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(data['Burned transformers 2020'])
plt.title('Distribución de la variable objetivo')
plt.xlabel('Transformadores quemados en 2020')
plt.ylabel('Frecuencia')
plt.show()