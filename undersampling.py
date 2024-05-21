import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import StandardScaler

# Leer el archivo Excel
excel_file = "./Dataset_Year_2020.xlsx"
data = pd.read_excel(excel_file)

# Comprobar las categorias unicas antes de la conversion
#print("Categorias unicas antes de la conversion:")
#print(data['Type of clients'].unique())
#print(data['Type of installation'].unique())

# Convertir las columnas categoricas en valores numericos (1 a N)
data['Type of clients'] = pd.factorize(data['Type of clients'])[0] + 1
data['Type of installation'] = pd.factorize(data['Type of installation'])[0] + 1

# Comprobar las categorias despues de la conversion
#print("Categorias despues de la conversion:")
#print(data['Type of clients'].unique())
#print(data['Type of installation'].unique())

# Definir las caracteristicas y la variable objetivo
X = data.drop(columns=['Burned transformers 2020'])
y = data['Burned transformers 2020'].astype(int)

# Dividir los datos en train y test (80% - 20%) con estratificacion
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Aplicar SMOTE-Tomek al train set
smote_tomek = SMOTETomek(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

# Aplicar estandarizacion al train set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_resampled)

# Entrenar el modelo SVM
svm_model = SVC(kernel='linear', class_weight='balanced', random_state=42)
svm_model.fit(X_train_scaled, y_resampled)

# Aplicar las mismas transformaciones al test set
X_test_scaled = scaler.transform(X_test)

# Predecir las etiquetas del test set
y_pred = svm_model.predict(X_test_scaled)

# Calcular y mostrar las metricas
f_score = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, svm_model.decision_function(X_test_scaled))

print("SVM + SMOTE Tomek")

print("F-score del modelo SVM:", f_score)
print("Accuracy del modelo SVM:", accuracy)
print("Recall del modelo SVM:", recall)
print("ROC AUC del modelo SVM:", roc_auc)
print("Matriz de Confusion:")
print(conf_matrix)
print("Reporte de Clasificacion:")
print(classification_rep)
