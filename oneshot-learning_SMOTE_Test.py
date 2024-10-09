import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, precision_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Leer el archivo Excel
excel_file = "./Dataset_Year_2020.xlsx"
data = pd.read_excel(excel_file)

# Definir las variables categóricas y la variable objetivo
categorical_columns = ['Type of clients', 'Type of installation']
target_column = 'Burned transformers 2020'

# Distribución de la variable objetivo
#plt.figure(figsize=(6,4))
#sns.countplot(x=target_column, data=data)
#plt.title("Distribución de la Variable Objetivo: 'Burned transformers 2020'")
#plt.xlabel("Burned transformers 2020")
#plt.ylabel("Count")
#plt.show()

# Separar las características (X) y la variable objetivo (y)
X = data[categorical_columns]
y = data[target_column]

# Codificar las variables categóricas usando OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded = encoder.fit_transform(X)

# Escalar los datos
scaler = StandardScaler()
X_encoded_scaled = scaler.fit_transform(X_encoded)

# Usar LabelEncoder para convertir la columna objetivo en números
le = LabelEncoder()
y = le.fit_transform(y)

# Aplicar SMOTE para oversampling (balancear las clases)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_encoded_scaled, y)

# Dividir en entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, stratify=y_resampled, random_state=42)

# Crear un modelo de red neuronal densa más simple
def create_simple_network(input_shape):
    model = models.Sequential()
    # Encapsulate input_shape in a tuple 
    model.add(layers.Input(shape=(input_shape,)))  # Changed this line
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dense(1, activation='sigmoid'))  # Usamos sigmoid para clasificación binaria
    return model

# Crear el modelo
input_shape = X_train.shape[1]
model = create_simple_network(input_shape)

# Compilar el modelo
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64)

# Evaluar el modelo en el conjunto de prueba
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy en el conjunto de prueba: {accuracy:.4f}")

# Calcular el AUC-ROC para interpretar el rendimiento
roc_auc = roc_auc_score(y_test, model.predict(X_test))
print(f"ROC-AUC en el conjunto de prueba: {roc_auc:.4f}")


# Predecir las clases en el conjunto de prueba
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Calcular la matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Mostrar la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión")
plt.show()