import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler

# Leer el archivo Excel
excel_file = "./Dataset_Year_2020.xlsx"
data = pd.read_excel(excel_file)

# Aplicar One-Hot Encoding a las columnas categóricas
data = pd.get_dummies(data, columns=['Type of clients', 'Type of installation'])

# Definir las características y la variable objetivo antes del oversampling
X = data.drop(columns=['Burned transformers 2020'])
y = data['Burned transformers 2020'].astype(int)

# Aplicar RandomOverSampler para oversampling
ros = RandomOverSampler(sampling_strategy='minority', random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

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

print(data.info())

# Imprimir el máximo, mínimo, media y desviación estándar de cada columna float64
for col in float_columns:
    max_value = data[col].max()
    min_value = data[col].min()
    mean_value = data[col].mean()
    std_value = data[col].std()
    print(f"Columna: {col}")
    print(f"  Máximo: {max_value}")
    print(f"  Media: {mean_value}")
    print(f"  Mínimo: {min_value}")
    print(f"  Desviación Estándar: {std_value}")
    print()