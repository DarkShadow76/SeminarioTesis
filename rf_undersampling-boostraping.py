import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix, roc_auc_score, precision_score, classification_report
import category_encoders as ce
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import randint

# Leer el archivo Excel
excel_file = "./Dataset_Year_2020.xlsx"
data = pd.read_excel(excel_file)

# Definir las variables categóricas y la variable objetivo
categorical_columns = ['Type of clients', 'Type of installation']
target_column = 'Burned transformers 2020'

# Elegir el método de codificación
encoding_method = 'target'

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba/validación (80%/20%)
X = data.drop(columns=[target_column])
y = data[target_column].astype(int)

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Definir la función para aplicar diferentes métodos de codificación
def encode_data(train_data, test_data, target_column, method=''):
    if method == 'one-hot':
        train_data = pd.get_dummies(train_data, columns=categorical_columns)
        test_data = pd.get_dummies(test_data, columns=categorical_columns)
        test_data = test_data.reindex(columns=train_data.columns, fill_value=0)
        return train_data, test_data, None
    elif method == 'label':
        label_encoders = {col: LabelEncoder() for col in categorical_columns}
        for col, le in label_encoders.items():
            train_data[col] = le.fit_transform(train_data[col])
            test_data[col] = le.transform(test_data[col])
        return train_data, test_data, None
    elif method == 'target':
        target_encoder = ce.TargetEncoder(cols=categorical_columns)
        train_data_encoded = target_encoder.fit_transform(train_data, train_data[target_column])
        test_data_encoded = target_encoder.transform(test_data)
        return train_data_encoded, test_data_encoded, target_encoder
    elif method == 'frequency':
        for col in categorical_columns:
            freq = train_data[col].value_counts(normalize=True)
            train_data[col] = train_data[col].map(freq)
            test_data[col] = test_data[col].map(freq).fillna(0)
        return train_data, test_data, None
    elif method == 'leave-one-out':
        loo_encoder = ce.LeaveOneOutEncoder(cols=categorical_columns)
        train_data_encoded = loo_encoder.fit_transform(train_data, train_data[target_column])
        test_data_encoded = loo_encoder.transform(test_data)
        return train_data_encoded, test_data_encoded, loo_encoder
    else:
        raise ValueError("Método de codificación no soportado")

# Añadir la columna objetivo temporalmente para la codificación
X_train_val[target_column] = y_train_val
X_test[target_column] = y_test

# Aplicar el método de codificación seleccionado
X_train_val, X_test, encoder = encode_data(X_train_val, X_test, target_column, method=encoding_method)

# Eliminar la columna objetivo después de la codificación
X_train_val = X_train_val.drop(columns=[target_column])
X_test = X_test.drop(columns=[target_column])

# Guardar el encoder entrenado en un archivo
if encoder is not None:
    encoder_path = "target_encoder.pkl"
    joblib.dump(encoder, encoder_path)
    print(f'Encoder guardado como {encoder_path}')

# Aplicar RandomUnderSampler para undersampling en los datos de entrenamiento
rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train_val, y_train_val)

# Definir parámetros para Random Forest
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]
}

# Validación cruzada con StratifiedKFold
skf = StratifiedKFold(n_splits=5)

# Definir Random Forest con búsqueda de hiperparámetros
rf = RandomForestClassifier(random_state=42)
rf_random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=50, cv=5, verbose=2, random_state=42, n_jobs=-1)

# Ajustar el modelo
rf_random_search.fit(X_train_resampled, y_train_resampled)

# Imprimir los mejores hiperparámetros
print(f"Mejores hiperparámetros encontrados: {rf_random_search.best_params_}")

# Evaluar el modelo con los mejores hiperparámetros en el conjunto de validación
rf_best = rf_random_search.best_estimator_
y_pred_rf = rf_best.predict(X_test)
y_prob_rf = rf_best.predict_proba(X_test)[:, 1]

# Métricas para Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)

print(f"Accuracy: {accuracy_rf:.4f}")
print(f"F1 Score: {f1_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print(f"ROC AUC: {roc_auc_rf:.4f}")

# Guardar el modelo entrenado en un archivo
joblib_file = "best_random_forest_model.pkl"  
joblib.dump(rf_best, joblib_file)
print(f'Modelo guardado como {joblib_file}')

# Aplicar AdaBoost
adaboost = AdaBoostClassifier(estimator=rf_best, n_estimators=50, learning_rate=1.0, random_state=42)
adaboost.fit(X_train_resampled, y_train_resampled)

# Evaluar AdaBoost en el conjunto de prueba
y_pred_adaboost = adaboost.predict(X_test)
y_prob_adaboost = adaboost.predict_proba(X_test)[:, 1]

# Métricas para AdaBoost
accuracy_adaboost = accuracy_score(y_test, y_pred_adaboost)
f1_adaboost = f1_score(y_test, y_pred_adaboost)
precision_adaboost = precision_score(y_test, y_pred_adaboost)
recall_adaboost = recall_score(y_test, y_pred_adaboost)
roc_auc_adaboost = roc_auc_score(y_test, y_prob_adaboost)

print("\nResultados de AdaBoost:")
print(f"Accuracy: {accuracy_adaboost:.4f}")
print(f"F1 Score: {f1_adaboost:.4f}")
print(f"Precision: {precision_adaboost:.4f}")
print(f"Recall: {recall_adaboost:.4f}")
print(f"ROC AUC: {roc_auc_adaboost:.4f}")

# Guardar el modelo de AdaBoost en un archivo
joblib_file_adaboost = "best_adaboost_model.pkl"  
joblib.dump(adaboost, joblib_file_adaboost)
print(f'Modelo de AdaBoost guardado como {joblib_file_adaboost}')
