import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix, roc_auc_score, precision_score, classification_report

# Leer el archivo Excel
excel_file = "./Dataset_Year_2020.xlsx"
data = pd.read_excel(excel_file)

# Aplicar One-Hot Encoding a las columnas categóricas
data = pd.get_dummies(data, columns=['Type of clients', 'Type of installation'])

# Definir las características y la variable objetivo antes del undersampling
X = data.drop(columns=['Burned transformers 2020'])
y = data['Burned transformers 2020'].astype(int)

# Variables binarias que no queremos estandarizar
binary_columns = ['LOCATION', 'SELF-PROTECTION', 'Criticality according to previous study for ceramics level', 
                  'Removable connectors', 'Air network', 'Circuit Queue', 'Burned transformers 2020']

# Variables no binarias y no dummy que queremos estandarizar
non_binary_columns = [col for col in data.columns if col 
                      not in binary_columns 
                      and not col.startswith('Type of clients_') 
                      and not col.startswith('Type of installation_')]

# Seleccionar las columnas que son de tipo float64
float_columns = data.select_dtypes(include=['float64']).columns

# Estandarizar las columnas no binarias
scaler = StandardScaler()
data[float_columns] = scaler.fit_transform(data[float_columns])

"""
# Aplicar SelectKBest para seleccionar las mejores características
selector = SelectKBest(chi2, k=10)  # Seleccionar las 10 mejores características
selector.fit(X, y)
X_selected = selector.transform(X)
"""

X_selected = X

# Diccionarios para acumular las métricas de los modelos
metrics = {
    'SVM': {'accuracy': [], 'precision': [], 'f1': [], 'recall': [], 'roc_auc': [], 'conf_matrix': [], 'classification_report': []},
    'Naive Bayes': {'accuracy': [], 'precision': [], 'f1': [], 'recall': [], 'roc_auc': [], 'conf_matrix': [], 'classification_report': []},
    'Random Forest': {'accuracy': [], 'precision': [], 'f1': [], 'recall': [], 'roc_auc': [], 'conf_matrix': [], 'classification_report': []}
}

# Realizar 100 iteraciones de muestreo aleatorio y entrenamiento
for i in range(50):
    # Aplicar RandomUnderSampler para undersampling
    rus = RandomUnderSampler(sampling_strategy={0: 503, 1: 503}, random_state=i)
    X_resampled, y_resampled = rus.fit_resample(X_selected, y)
    
    # Separar los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
    
    # Entrenar y evaluar Random Forest
    rf = RandomForestClassifier(random_state=42)
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
for model_name, model_metrics in metrics.items():
    print(f"\nMétricas promedio para el modelo {model_name}")
    for metric_name, metric_values in model_metrics.items():
        if metric_name == 'conf_matrix':
            print(f"Matriz de Confusión Promedio:\n{mean_conf_matrix(metric_values)}")
        elif metric_name == 'classification_report':
            avg_report = mean_classification_report(metric_values)
            print("Reporte de Clasificación Promedio:")
            for label, metrics in avg_report.items():
                print(f"Clase {label}:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")
        else:
            print(f"{metric_name.capitalize()}: {np.mean(metric_values):.4f} ± {np.std(metric_values):.4f}")