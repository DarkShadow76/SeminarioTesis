import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix, roc_auc_score, precision_score, classification_report
import category_encoders as ce

# Leer el archivo Excel
excel_file = "./Dataset_Year_2020.xlsx"
data = pd.read_excel(excel_file)

# Definir las variables categóricas y la variable objetivo
categorical_columns = ['Type of clients', 'Type of installation']
target_column = 'Burned transformers 2020'

# Elegir el método de codificación
encoding_method = 'leave-one-out'  # Cambia esto a 'label', 'target', 'frequency', 'leave-one-out'

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba/validación (80%/20%)
X = data.drop(columns=[target_column])
y = data[target_column].astype(int)

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Definir la función para aplicar diferentes métodos de codificación
def encode_data(train_data, test_data, target_column, method=''):
    if method == 'one-hot':
        train_data = pd.get_dummies(train_data, columns=categorical_columns)
        test_data = pd.get_dummies(test_data, columns=categorical_columns)
        # Asegurar que ambas tengan las mismas columnas
        test_data = test_data.reindex(columns=train_data.columns, fill_value=0)
        return train_data, test_data
    elif method == 'label':
        label_encoders = {col: LabelEncoder() for col in categorical_columns}
        for col, le in label_encoders.items():
            train_data[col] = le.fit_transform(train_data[col])
            test_data[col] = le.transform(test_data[col])
        return train_data, test_data
    elif method == 'target':
        target_encoder = ce.TargetEncoder(cols=categorical_columns)
        train_data = target_encoder.fit_transform(train_data, train_data[target_column])
        test_data = target_encoder.transform(test_data)
        return train_data, test_data
    elif method == 'frequency':
        for col in categorical_columns:
            freq = train_data[col].value_counts(normalize=True)
            train_data[col] = train_data[col].map(freq)
            test_data[col] = test_data[col].map(freq).fillna(0)
        return train_data, test_data
    elif method == 'leave-one-out':
        loo_encoder = ce.LeaveOneOutEncoder(cols=categorical_columns)
        train_data = loo_encoder.fit_transform(train_data, train_data[target_column])
        test_data = loo_encoder.transform(test_data)
        return train_data, test_data
    else:
        raise ValueError("Método de codificación no soportado")

# Añadir la columna objetivo temporalmente para la codificación
X_train_val[target_column] = y_train_val
X_test[target_column] = y_test

# Aplicar el método de codificación seleccionado
X_train_val, X_test = encode_data(X_train_val, X_test, target_column, method=encoding_method)

# Eliminar la columna objetivo después de la codificación
X_train_val = X_train_val.drop(columns=[target_column])
X_test = X_test.drop(columns=[target_column])

# Variables binarias que no queremos estandarizar
binary_columns = ['LOCATION', 'SELF-PROTECTION', 'Criticality according to previous study for ceramics level', 
                  'Removable connectors', 'Air network', 'Circuit Queue', 'Burned transformers 2020']

# Variables no binarias y no dummy que queremos estandarizar
non_binary_columns = [col for col in X_train_val.columns if col 
                      not in binary_columns 
                      and not col.startswith('Type of clients_') 
                      and not col.startswith('Type of installation_')]

# Eliminar columnas específicas del dataset original
columns_to_drop = [
]

X_train_val = X_train_val.drop(columns=columns_to_drop)
X_test = X_test.drop(columns=columns_to_drop)

# Aplicar RandomUnderSampler para undersampling en los datos de entrenamiento
rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train_val, y_train_val)

# Validación cruzada con StratifiedKFold
skf = StratifiedKFold(n_splits=10)

# Diccionario para acumular las métricas del modelo
metrics = {
    'Random Forest': {'accuracy': [], 'precision': [], 'f1': [], 'recall': [], 'roc_auc': [], 'conf_matrix': [], 'classification_report': []}
}

# Realizar la validación cruzada
for i, (train_index, val_index) in enumerate(skf.split(X_train_resampled, y_train_resampled)):
    X_train_fold, X_val_fold = X_train_resampled.iloc[train_index], X_train_resampled.iloc[val_index]
    y_train_fold, y_val_fold = y_train_resampled.iloc[train_index], y_train_resampled.iloc[val_index]
    
    # Entrenar y evaluar Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_fold, y_train_fold)
    y_pred_rf = rf.predict(X_val_fold)
    y_prob_rf = rf.predict_proba(X_val_fold)[:, 1]

    metrics['Random Forest']['accuracy'].append(accuracy_score(y_val_fold, y_pred_rf))
    metrics['Random Forest']['precision'].append(precision_score(y_val_fold, y_pred_rf))
    metrics['Random Forest']['f1'].append(f1_score(y_val_fold, y_pred_rf))
    metrics['Random Forest']['recall'].append(recall_score(y_val_fold, y_pred_rf))
    metrics['Random Forest']['roc_auc'].append(roc_auc_score(y_val_fold, y_prob_rf))
    metrics['Random Forest']['conf_matrix'].append(confusion_matrix(y_val_fold, y_pred_rf))
    metrics['Random Forest']['classification_report'].append(classification_report(y_val_fold, y_pred_rf, output_dict=True))

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
