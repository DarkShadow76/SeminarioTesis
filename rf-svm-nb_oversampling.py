import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix, classification_report, roc_auc_score

# Leer el archivo Excel
excel_file = "./Dataset_Year_2020.xlsx"
data = pd.read_excel(excel_file)

# Aplicar One-Hot Encoding a las columnas categóricas
data = pd.get_dummies(data, columns=['Type of clients', 'Type of installation'])

# Definir las características y la variable objetivo antes del oversampling
X = data.drop(columns=['Burned transformers 2020'])
y = data['Burned transformers 2020'].astype(int)

# Definir la cantidad de muestras adicionales que quieres generar
desired_samples = 10000

# Calcular la proporción correspondiente
total_minority_class = y.value_counts()[1] + desired_samples
sampling_strategy = total_minority_class / y.value_counts()[0]

# Aplicar SMOTE para oversampling
smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

"""
# Aplicar ADASYN para oversampling
adasyn = ADASYN(sampling_strategy='minority', random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X, y)

# Aplicar SMOTE para oversampling
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Aplicar RandomOverSampler para oversampling
ros = RandomOverSampler(sampling_strategy='minority', random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)
"""
# Actualizar el DataFrame con los datos oversampleados
data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
                  pd.DataFrame(y_resampled, columns=['Burned transformers 2020'])], axis=1)

print("Conteo de 'Burned transformers 2020' Después del oversampling")
print(data['Burned transformers 2020'].value_counts())

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

# Tratar valores outliers

# Separar los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['Burned transformers 2020']), data['Burned transformers 2020'], test_size=0.3, random_state=42)

# Entrenar y evaluar SVM
svm = SVC(probability=True)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
y_prob_svm = svm.predict_proba(X_test)[:, 1]

f_score_svm = f1_score(y_test, y_pred_svm)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
roc_auc_svm = roc_auc_score(y_test, y_prob_svm)

print("\nMetricas para el modelo SVM")
print("F-score:", f_score_svm)
print("Exactitud:", accuracy_svm)
print("Recall:", recall_svm)
print("Matriz de Confusión:")
print(conf_matrix_svm)
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred_svm))
print("AUC:", roc_auc_svm)

# Entrenar y evaluar Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
y_prob_nb = nb.predict_proba(X_test)[:, 1]

f_score_nb = f1_score(y_test, y_pred_nb)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
recall_nb = recall_score(y_test, y_pred_nb)
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
roc_auc_nb = roc_auc_score(y_test, y_prob_nb)

print("\nMetricas para el modelo Naive Bayes")
print("F-score:", f_score_nb)
print("Exactitud:", accuracy_nb)
print("Recall:", recall_nb)
print("Matriz de Confusión:")
print(conf_matrix_nb)
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred_nb))
print("AUC:", roc_auc_nb)

# Entrenar y evaluar Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

f_score_rf = f1_score(y_test, y_pred_rf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)

print("\nMetricas para el modelo Random Forest")
print("F-score:", f_score_rf)
print("Exactitud:", accuracy_rf)
print("Recall:", recall_rf)
print("Matriz de Confusión:")
print(conf_matrix_rf)
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred_rf))
print("AUC:", roc_auc_rf)