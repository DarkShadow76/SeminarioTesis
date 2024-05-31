import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import NearMiss

# Leer el archivo Excel
excel_file = "./Dataset_Year_2020.xlsx"
data = pd.read_excel(excel_file)

# Aplicar One-Hot Encoding a las columnas categóricas
data = pd.get_dummies(data, columns=['Type of clients', 'Type of installation'])

# Definir las caracteristicas y la variable objetivo
X = data.drop(columns=['Burned transformers 2020'])
y = data['Burned transformers 2020'].astype(int)

# Dividir los datos en train y test (80% - 20%) con estratificacion
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Aplicar NearMiss para undersampling en el conjunto de entrenamiento
nearmiss = NearMiss(sampling_strategy='majority')
X_train_resampled, y_train_resampled = nearmiss.fit_resample(X_train, y_train)

# Imprimir los conteos de la variable objetivo después del undersampling
print("Conteo de 'Burned transformers 2020' despues del undersampling en conjunto de entrenamiento:")
print(y_train_resampled.value_counts())

# Estandarizar los datos resampleados y de prueba
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Entrenar el modelo SVM con los datos balanceados
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_resampled, y_train_resampled)

# Predecir las etiquetas del conjunto de prueba
y_pred = svm_model.predict(X_test)

# Calcular las métricas
f_score = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print("SVM + NearMiss")

# Imprimir las métricas
print("F-score del modelo SVM:", f_score)
print("Exactitud del modelo SVM:", accuracy)
print("Recall del modelo SVM:", recall)
print("Matriz de Confusion:")
print(conf_matrix)
print("Reporte de Clasificacion:")
print(classification_report(y_test, y_pred))
print("AUC del modelo SVM:", roc_auc)