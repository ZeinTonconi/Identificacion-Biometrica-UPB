import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from deepface import DeepFace
import sys
import logging

# Configurar consola para UTF-8 para evitar UnicodeEncodeError
sys.stdout.reconfigure(encoding='utf-8')

# Configurar logging para guardar errores en un archivo
logging.basicConfig(filename='deepface_errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Carpeta donde guardaste los datos
data_dir = 'csv_integrado/dataUnificado'

# --- CARGAR DATOS ---
X_all, y_all = [], []

for file in os.listdir(data_dir):
    if file.startswith('faces_') and file.endswith('.pkl'):
        name = file.replace('faces_', '').replace('.pkl', '')
        try:
            faces = pickle.load(open(os.path.join(data_dir, file), 'rb'))
        except Exception as e:
            logging.error(f"Error loading {file}: {str(e)}")
            continue
        
        embeddings = []
        for idx, face in enumerate(faces):
            try:
                # Convertir de [0,1] a [0,255] y formato imagen
                face_img = (face.reshape(100, 100) * 255).astype(np.uint8)
                face_img = np.stack([face_img] * 3, axis=-1)  # Convertir a BGR
                # Extraer embedding con Facenet512
                embedding = DeepFace.represent(face_img, model_name='Facenet512', enforce_detection=False)
                embeddings.append(embedding[0]['embedding'])
            except Exception as e:
                logging.error(f"Error procesando embedding para {name} (imagen {idx}): {str(e)}")
                continue
        if embeddings:  # Solo añadir si hay embeddings válidos
            X_all.extend(embeddings)
            y_all.extend([name] * len(embeddings))
            print(f"Procesado {name}: {len(embeddings)} embeddings válidos")
        else:
            logging.warning(f"No se generaron embeddings para {name}")

X_all = np.array(X_all)
y_all = np.array(y_all)

if len(X_all) == 0:
    raise ValueError("No se generaron embeddings válidos. Revisa el log 'deepface_errors.log'.")

print(f"Total data: {X_all.shape}, total labels: {len(y_all)}")
print(f"Number of unique classes: {len(np.unique(y_all))}")
# Verificar balance de clases
unique, counts = np.unique(y_all, return_counts=True)
print("Distribución de clases:", dict(zip(unique, counts)))

# --- DIVIDIR EN TRAIN Y TEST ---
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# --- ENTRENAR SVM CON GRIDSEARCH ---
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto']
}
model = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
model.fit(X_train, y_train)

print(f"Mejores parámetros: {model.best_params_}")
print(f"Mejor cross-val accuracy: {model.best_score_:.3f}")

# --- PREDICCIONES ---
y_pred = model.predict(X_test)

# --- MÉTRICAS ---
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

print(f"Accuracy: {acc:.3f}")
print(f"Precision (macro): {prec:.3f}")
print(f"Recall (macro): {rec:.3f}")
print(f"F1-score (macro): {f1:.3f}")

# --- Fallback a KNN si SVM no es suficiente ---
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)  # k=1 para maximizar precisión
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
print(f"KNN Accuracy (k=1): {acc_knn:.3f}")