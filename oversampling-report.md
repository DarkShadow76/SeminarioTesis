a```markdown
Conteo de 'Burned transformers 2020' después del undersampling en conjunto de entrenamiento:
1    14195
0    12195
Name: Burned transformers 2020, dtype: int64

### Métricas para el modelo SVM:
- F-score: 0.11811023622047244
- Exactitud: 0.647244094488189
- Recall: 0.5952380952380952

#### Matriz de Confusión:
|               | Predicción No Quemado | Predicción Quemado |
|---------------|------------------------|---------------------|
| Real No Quemado | 1980                   | 1069                |
| Real Quemado     | 51                     | 75                  |

- Reporte de Clasificación:
  ```
              precision    recall  f1-score   support

           0       0.97      0.65      0.78      3049
           1       0.07      0.60      0.12       126

    accuracy                           0.65      3175
   macro avg       0.52      0.62      0.45      3175
weighted avg       0.94      0.65      0.75      3175
  ```

- AUC: 0.6223156694622749

### Métricas para el modelo Naive Bayes:
- F-score: 0.089540412044374
- Exactitud: 0.2762204724409449
- Recall: 0.8968253968253969

#### Matriz de Confusión:
|               | Predicción No Quemado | Predicción Quemado |
|---------------|------------------------|---------------------|
| Real No Quemado | 764                    | 2285                |
| Real Quemado     | 13                     | 113                 |

- Reporte de Clasificación:
  ```
              precision    recall  f1-score   support

           0       0.98      0.25      0.40      3049
           1       0.05      0.90      0.09       126

    accuracy                           0.28      3175
   macro avg       0.52      0.57      0.24      3175
weighted avg       0.95      0.28      0.39      3175
  ```

- AUC: 0.5736996777501862

### Métricas para el modelo Random Forest:
- F-score: 0.2094240837696335
- Exactitud: 0.9524409448818898
- Recall: 0.15873015873015872

#### Matriz de Confusión:
|               | Predicción No Quemado | Predicción Quemado |
|---------------|------------------------|---------------------|
| Real No Quemado | 3004                   | 45                  |
| Real Quemado     | 106                    | 20                  |

- Reporte de Clasificación:
  ```
              precision    recall  f1-score   support

           0       0.97      0.99      0.98      3049
           1       0.31      0.16      0.21       126

    accuracy                           0.95      3175
   macro avg       0.64      0.57      0.59      3175
weighted avg       0.94      0.95      0.95      3175
  ```

- AUC: 0.5719856106868242
```
