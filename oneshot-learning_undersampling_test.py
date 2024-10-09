import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE  # Añadido SMOTE para balanceo
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix, roc_auc_score, precision_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Leer el archivo Excel
excel_file = "./Dataset_Year_2020.xlsx"
data = pd.read_excel(excel_file)

# Definir las variables categóricas y la variable objetivo
categorical_columns = ['Type of clients', 'Type of installation']
target_column = 'Burned transformers 2020'

# Distribución de la variable objetivo
plt.figure(figsize=(6,4))
sns.countplot(x=target_column, data=data)
plt.title("Distribución de la Variable Objetivo: 'Burned transformers 2020'")
plt.xlabel("Burned transformers 2020")
plt.ylabel("Count")
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

# Definir la red siamesa para One-shot learning con regularización
def create_siamese_network(input_shape):
    base_network = models.Sequential()
    base_network.add(layers.Input(shape=input_shape))
    base_network.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    base_network.add(layers.Dropout(0.5))  # Dropout para regularización
    base_network.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    base_network.add(layers.Dropout(0.5))
    base_network.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    base_network.add(layers.Dropout(0.5))
    base_network.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    base_network.add(layers.Dense(16, activation='relu'))
    base_network.add(layers.Dense(1, activation='sigmoid'))  # Cambiado a sigmoid para clasificación binaria
    return base_network

# Función para calcular la distancia euclidiana
def euclidean_distance(vectors):
    x, y = vectors
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))

# Definir las entradas de la red siamesa
input_shape = X_train.shape[1:]

input_a = layers.Input(shape=input_shape)
input_b = layers.Input(shape=input_shape)

# Compartir la red base
base_network = create_siamese_network(input_shape)

processed_a = base_network(input_a)
processed_b = base_network(input_b)

# Calcular la distancia entre las dos entradas
distance = layers.Lambda(euclidean_distance)([processed_a, processed_b])

# Definir el modelo de la red siamesa
model = models.Model([input_a, input_b], distance)

# Definir la pérdida focal para manejar el desbalanceo
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1)) - tf.reduce_sum((1 - alpha) * tf.pow( pt_0, gamma) * tf.math.log(1. - pt_0))
    return focal_loss_fixed

# Compilar el modelo con focal loss y un optimizador ajustado
model.compile(loss=focal_loss(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

# Imprimir la estructura del modelo
model.summary()

# Generar pares de entrenamiento
def make_pairs(X, y):
    pairs = []
    labels = []
    num_classes = len(np.unique(y))

    # Crear listas de ejemplos por clase
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

    for idx in range(len(X)):
        x1 = X[idx]
        label = y[idx]

        # Elegir un ejemplo de la misma clase
        positive_idx = np.random.choice(digit_indices[label])
        x2 = X[positive_idx]

        pairs += [[x1, x2]]
        labels += [1]

        # Elegir un ejemplo de una clase diferente
        negative_label = np.random.choice([l for l in range(num_classes) if l != label])
        negative_idx = np.random.choice(digit_indices[negative_label])
        x2 = X[negative_idx]

        pairs += [[x1, x2]]
        labels += [0]

    return np.array(pairs), np.array(labels)

# Generar pares de entrenamiento
pairs_train, labels_train = make_pairs(X_train, y_train)
pairs_test, labels_test = make_pairs(X_test, y_test)

# Entrenar el modelo
history = model.fit(
    [pairs_train[:, 0], pairs_train[:, 1]], labels_train,
    validation_data=([pairs_test[:, 0], pairs_test[:, 1]], labels_test),
    epochs=50,  # Aumentadas las épocas para un mejor entrenamiento
    batch_size=64  # Tamaño de batch ajustado
)

# Evaluar el modelo en el conjunto de prueba
loss, accuracy = model.evaluate([pairs_test[:, 0], pairs_test[:, 1]], labels_test)

print(f"Accuracy en el conjunto de prueba: {accuracy:.4f}")

# Evaluar el AUC-ROC para mejor interpretación del rendimiento en clases desbalanceadas
roc_auc = roc_auc_score(labels_test, model.predict([pairs_test[:, 0], pairs_test[:, 1]]))
print(f"ROC-AUC en el conjunto de prueba: {roc_auc:.4f}")
