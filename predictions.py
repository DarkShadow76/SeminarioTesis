import pandas as pd
import numpy as np
import category_encoders as ce
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Leer el archivo Excel para 2019
excel_file_2019 = "./Dataset_Year_2019.xlsx"
data_2019 = pd.read_excel(excel_file_2019)

# Definir las variables categóricas y la variable objetivo
categorical_columns = ['Type of clients', 'Type of installation']
target_column_2019 = 'Burned transformers 2019'

# Elegir el método de codificación
encoding_method = 'target'  # Debe coincidir con el método usado para entrenar el modelo

# Cargar el modelo y el encoder
model_file = "random_forest_model.pkl"
encoder_file = "target_encoder.pkl"

# Cargar el modelo RandomForest y el TargetEncoder
rf = joblib.load(model_file)
target_encoder = joblib.load(encoder_file)

# Separar las características y la variable objetivo
X_2019 = data_2019[categorical_columns]
y_true_2019 = data_2019[target_column_2019]

# Verificar las columnas esperadas por el encoder
expected_columns = target_encoder.get_feature_names_out()
# Asegurarse de que X_2019 tenga solo las columnas esperadas
X_2019 = X_2019.reindex(columns=expected_columns, fill_value=0)

# Aplicar la codificación
X_2019_encoded = target_encoder.transform(X_2019)

# Realizar las predicciones
y_pred_2019 = rf.predict(X_2019_encoded)

# Evaluar las métricas
accuracy = accuracy_score(y_true_2019, y_pred_2019)
precision = precision_score(y_true_2019, y_pred_2019, average='weighted')
recall = recall_score(y_true_2019, y_pred_2019, average='weighted')
conf_matrix = confusion_matrix(y_true_2019, y_pred_2019)

# Imprimir las métricas
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Matriz de Confusión:\n{conf_matrix}")

# Guardar las predicciones en un archivo Excel (opcional)
predictions = pd.DataFrame({
    'True Values': y_true_2019,
    'Predictions': y_pred_2019
})

predictions.to_excel("predictions_2019.xlsx", index=False)
print("Predicciones guardadas en predictions_2019.xlsx")
