import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Carpeta donde guardaste los datos
data_dir = 'csv_integrado/dataUnificado'

# --- CARGAR DATOS ---
X_all, y_all = [], []

for file in os.listdir(data_dir):
    if file.startswith('faces_') and file.endswith('.pkl'):
        name = file.replace('faces_', '').replace('.pkl', '')
        faces = pickle.load(open(os.path.join(data_dir, file), 'rb'))
        X_all.extend(faces)
        y_all.extend([name] * len(faces))

X_all = np.array(X_all)
y_all = np.array(y_all)

print(f"Total data: {X_all.shape}, total labels: {len(y_all)}")

# --- DIVIDIR EN TRAIN Y TEST ---
# Usamos stratify para mantener balance por persona
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.85, random_state=42, stratify=y_all
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# --- ENTRENAR KNN ---
knn = KNeighborsClassifier(n_neighbors=7)  # Ajusta n_neighbors si quieres
knn.fit(X_all, y_all)

# --- PREDICCIONES ---
y_pred = knn.predict(X_test)

# --- MÉTRICAS ---
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

print(f"Accuracy: {acc:.3f}")
print(f"Precision (macro): {prec:.3f}")
print(f"Recall (macro): {rec:.3f}")
print(f"F1-score (macro): {f1:.3f}")