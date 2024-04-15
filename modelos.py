from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.cluster import KMeans
import numpy as np

# Cargar el conjunto de datos de cáncer de mama de Wisconsin
data = load_breast_cancer()
X = data.data  # características
y = data.target  # etiquetas

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=600)

# Crear un modelo de clasificación de bosques aleatorios
rf = RandomForestClassifier(n_estimators=100, random_state=600)

# Entrenar el modelo con el conjunto de entrenamiento
rf.fit(X_train, y_train)

# Predecir las etiquetas para el conjunto de prueba
y_pred = rf.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy}')

# Calcular el porcentaje de tumores malignos
num_malignos = sum(y_pred)
total_tumores = len(y_pred)
porcentaje_malignos = (num_malignos / total_tumores) * 100
print(f'MODELO SUPERVISADO: Porcentaje de tumores malignos: {porcentaje_malignos}%')

#--------------------------------------------------------------------------------------------------------------------------------------------
# Crear un modelo de clustering KMeans con 2 clusters (benigno y maligno)
kmeans = KMeans(n_clusters=2, random_state=42)  

# Agrupar los datos en clusters
clusters = kmeans.fit_predict(X)

# Calcular el porcentaje de tumores malignos
benigno = np.where(clusters == 0)[0]
maligno = np.where(clusters == 1)[0]
total_tumores = len(clusters)
porcentaje_malignos = (len(maligno) / total_tumores) * 100
print(f'MODELO NO SUPERVISADO: Porcentaje de tumores malignos: {porcentaje_malignos}%')