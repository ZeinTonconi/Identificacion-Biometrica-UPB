import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from deepface import DeepFace
import sys
import logging

# Configuración
sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(filename='deepface_errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

data_dir = 'backend2/dataUnificado'

# --- CARGAR DATOS ---
X_all, y_all = [], []
faces_all = []  # opcional: guardar crop de imagen para referencia

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
                # Convertir de [0,1] a [0,255]
                face_img = (face.reshape(100, 100) * 255).astype(np.uint8)
                face_img = np.stack([face_img]*3, axis=-1)
                
                embedding = DeepFace.represent(face_img, model_name='Facenet512', enforce_detection=False)
                embeddings.append(embedding[0]['embedding'])
                faces_all.append(face)  # opcional
            except Exception as e:
                logging.error(f"Error procesando embedding para {name} (imagen {idx}): {str(e)}")
                continue

        if embeddings:
            X_all.extend(embeddings)
            y_all.extend([name]*len(embeddings))
            print(f"Procesado {name}: {len(embeddings)} embeddings válidos")
        else:
            logging.warning(f"No se generaron embeddings para {name}")

X_all = np.array(X_all)
y_all = np.array(y_all)
faces_all = np.array(faces_all)  # opcional

if len(X_all) == 0:
    raise ValueError("No se generaron embeddings válidos. Revisa el log 'deepface_errors.log'.")

# --- Guardar embeddings y labels en .pkl ---
os.makedirs("embeddings_data", exist_ok=True)
save_path = os.path.join("embeddings_data", "faces_dataset.pkl")
with open(save_path, "wb") as f:
    pickle.dump({"embeddings": X_all, "labels": y_all, "faces": faces_all}, f)
print(f"Embeddings guardados en: {save_path}")

# --- ENTRENAR KNN ---
param_grid = {
    'n_neighbors': [5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

knn = KNeighborsClassifier()
model = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
model.fit(X_all, y_all)

print(f"Mejores parámetros: {model.best_params_}")
print(f"Mejor cross-val accuracy: {model.best_score_:.3f}")

# --- PREDICCIONES ---
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)
y_pred = model.predict(X_test)

# --- MÉTRICAS ---
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")

# --- Guardar modelo KNN ---
os.makedirs("modelos", exist_ok=True)
model_path = os.path.join("modelos", "knn_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model.best_estimator_, f)
print(f"Modelo guardado en: {model_path}")
