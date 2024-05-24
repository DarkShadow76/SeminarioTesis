import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import NearMiss

# Leer el archivo Excel
excel_file = "./Dataset_Year_2020.xlsx"
data = pd.read_excel(excel_file)

# Convertir las columnas categoricas en valores numericos (1 a N)
data['Type of clients'] = pd.factorize(data['Type of clients'])[0] + 1
data['Type of installation'] = pd.factorize(data['Type of installation'])[0] + 1

# Definir las caracteristicas y la variable objetivo
X = data.drop(columns=['Burned transformers 2020'])
y = data['Burned transformers 2020'].astype(int)

# Dividir los datos en train y test (80% - 20%) con estratificacion
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Aplicar NearMiss para undersampling
nearmiss = NearMiss(sampling_strategy='majority')
X_train_resampled, y_train_resampled = nearmiss.fit_resample(X_train, y_train)

# Imprimir los conteos de la variable objetivo después del undersampling
print("Conteo de 'Burned transformers 2020' despues del undersampling en conjunto de entrenamiento:")
print(y_train_resampled.value_counts())

# Estandarizar los datos resampleados
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Entrenar el modelo SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_resampled, y_train_resampled)

# Entrenar el modelo Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train_resampled, y_train_resampled)

# Entrenar el modelo Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

# Predecir las etiquetas del conjunto de prueba para SVM
y_pred_svm = svm_model.predict(X_test)

# Calcular las métricas para SVM
f_score_svm = f1_score(y_test, y_pred_svm)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
roc_auc_svm = roc_auc_score(y_test, y_pred_svm)

# Imprimir las métricas para SVM
print("Metricas para el modelo SVM:")
print("F-score:", f_score_svm)
print("Exactitud:", accuracy_svm)
print("Recall:", recall_svm)
print("Matriz de Confusión:")
print(conf_matrix_svm)
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred_svm))
print("AUC:", roc_auc_svm)

# Predecir las etiquetas del conjunto de prueba para Naive Bayes
y_pred_nb = nb_model.predict(X_test)

# Calcular las métricas para Naive Bayes
f_score_nb = f1_score(y_test, y_pred_nb)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
recall_nb = recall_score(y_test, y_pred_nb)
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
roc_auc_nb = roc_auc_score(y_test, y_pred_nb)

# Imprimir las métricas para Naive Bayes
print("\nMetricas para el modelo Naive Bayes:")
print("F-score:", f_score_nb)
print("Exactitud:", accuracy_nb)
print("Recall:", recall_nb)
print("Matriz de Confusión:")
print(conf_matrix_nb)
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred_nb))
print("AUC:", roc_auc_nb)

# Predecir las etiquetas del conjunto de prueba para Random Forest
y_pred_rf = rf_model.predict(X_test)

# Calcular las metricas para Random Forest
f_score_rf = f1_score(y_test, y_pred_rf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_rf)

# Imprimir las metricas para Random Forest
print("\nMetricas para el modelo Random Forest:")
print("F-score:", f_score_rf)
print("Exactitud:", accuracy_rf)
print("Recall:", recall_rf)
print("Matriz de Confusión:")
print(conf_matrix_rf)
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred_rf))
print("AUC:", roc_auc_rf)