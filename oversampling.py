import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import BorderlineSMOTE

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

# Aplicar Borderline SMOTE para oversampling
borderline_smote = BorderlineSMOTE(sampling_strategy='minority')
X_train_resampled, y_train_resampled = borderline_smote.fit_resample(X_train, y_train)

# Aplicar Borderline SMOTE para oversampling
borderline_smote = BorderlineSMOTE(sampling_strategy='minority')
X_train_resampled, y_train_resampled = borderline_smote.fit_resample(X_train, y_train)

# Estandarizar los datos resampleados
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Entrenar el modelo SVM
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

print("SVM + BorderLine SMOTE")

# Imprimir las métricas
print("F-score del modelo SVM:", f_score)
print("Exactitud del modelo SVM:", accuracy)
print("Recall del modelo SVM:", recall)
print("Matriz de Confusion:")
print(conf_matrix)
print("Reporte de Clasificacion:")
print(classification_report(y_test, y_pred))
print("AUC del modelo SVM:", roc_auc)
