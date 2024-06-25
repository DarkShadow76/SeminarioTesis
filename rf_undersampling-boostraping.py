import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix, roc_auc_score, precision_score, classification_report

# Leer el archivo Excel
excel_file = "./Dataset_Year_2020.xlsx"
data = pd.read_excel(excel_file)

# Aplicar One-Hot Encoding a las columnas categóricas
data = pd.get_dummies(data, columns=['Type of clients', 'Type of installation'])

# Variables binarias que no queremos estandarizar
binary_columns = ['LOCATION', 'SELF-PROTECTION', 'Criticality according to previous study for ceramics level', 
                  'Removable connectors', 'Air network', 'Circuit Queue', 'Burned transformers 2020']

# Variables no binarias y no dummy que queremos estandarizar
non_binary_columns = [col for col in data.columns if col 
                      not in binary_columns 
                      and not col.startswith('Type of clients_') 
                      and not col.startswith('Type of installation_')]

"""
# Identificar y eliminar outliers usando IQR
Q1 = data[non_binary_columns].quantile(0.25)
Q3 = data[non_binary_columns].quantile(0.75)
IQR = Q3 - Q1
data = data[~((data[non_binary_columns] < (Q1 - 1.5 * IQR)) | (data[non_binary_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]
"""

# Eliminar columnas específicas del dataset original
# Aquí coloca los nombres de las columnas que quieres eliminar
columns_to_drop = [
    'Type of installation_TORRE METALICA',
    'Type of installation_POLE WITH ANTI-FRAU NET',
    'Type of installation_POLE',
    'Type of installation_PAD MOUNTED',
    'Type of installation_OTROS',
    'Type of clients_STRATUM 6',
    'Type of clients_STRATUM 5',
    'Type of clients_STRATUM 4',
    'Type of clients_STRATUM 3',
    'Type of clients_STRATUM 2',
    'Type of clients_STRATUM 1'
]

data = data.drop(columns=columns_to_drop)

# Definir las características y la variable objetivo después de eliminar columnas
X = data.drop(columns=['Burned transformers 2020'])
y = data['Burned transformers 2020'].astype(int)

# Diccionario para acumular las métricas del modelo
metrics = {
    'Random Forest': {'accuracy': [], 'precision': [], 'f1': [], 'recall': [], 'roc_auc': [], 'conf_matrix': [], 'classification_report': []}
}

# Realizar 100 iteraciones de muestreo aleatorio y entrenamiento
for i in range(100):
    # Aplicar RandomUnderSampler para undersampling en los datos
    rus = RandomUnderSampler(sampling_strategy={0: 503, 1: 503}, random_state=i)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    
    # Separar los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=i)
    
    # Entrenar y evaluar Random Forest
    rf = RandomForestClassifier(random_state=i)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]

    metrics['Random Forest']['accuracy'].append(accuracy_score(y_test, y_pred_rf))
    metrics['Random Forest']['precision'].append(precision_score(y_test, y_pred_rf))
    metrics['Random Forest']['f1'].append(f1_score(y_test, y_pred_rf))
    metrics['Random Forest']['recall'].append(recall_score(y_test, y_pred_rf))
    metrics['Random Forest']['roc_auc'].append(roc_auc_score(y_test, y_prob_rf))
    metrics['Random Forest']['conf_matrix'].append(confusion_matrix(y_test, y_pred_rf))
    metrics['Random Forest']['classification_report'].append(classification_report(y_test, y_pred_rf, output_dict=True))

# Función para calcular la matriz de confusión promedio
def mean_conf_matrix(conf_matrix_list):
    sum_matrix = np.sum(conf_matrix_list, axis=0)
    return sum_matrix / len(conf_matrix_list)

# Función para calcular el reporte de clasificación promedio
def mean_classification_report(reports_list):
    avg_report = {}
    for label in reports_list[0].keys():
        if not isinstance(reports_list[0][label], dict):
            continue
        avg_report[label] = {}
        for metric in reports_list[0][label].keys():
            avg_report[label][metric] = np.mean([report[label][metric] for report in reports_list if isinstance(report[label], dict)])
    return avg_report

# Imprimir el reporte promedio de las métricas
model_name = 'Random Forest'
model_metrics = metrics[model_name]

print(f"\nMétricas promedio para el modelo {model_name}")
print(f"Accuracy: {np.mean(model_metrics['accuracy']):.4f} ± {np.std(model_metrics['accuracy']):.4f}")
print(f"Precision: {np.mean(model_metrics['precision']):.4f} ± {np.std(model_metrics['precision']):.4f}")
print(f"F1: {np.mean(model_metrics['f1']):.4f} ± {np.std(model_metrics['f1']):.4f}")
print(f"Recall: {np.mean(model_metrics['recall']):.4f} ± {np.std(model_metrics['recall']):.4f}")
print(f"Roc_auc: {np.mean(model_metrics['roc_auc']):.4f} ± {np.std(model_metrics['roc_auc']):.4f}")
print(f"Matriz de Confusión Promedio:\n{mean_conf_matrix(model_metrics['conf_matrix'])}")
avg_report = mean_classification_report(model_metrics['classification_report'])
print("Reporte de Clasificación Promedio:")
for label, metrics in avg_report.items():
    print(f"Clase {label}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
