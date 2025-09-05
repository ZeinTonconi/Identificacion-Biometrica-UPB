from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import pickle
import mediapipe as mp
import io
from PIL import Image
import os
from deepface import DeepFace
from sklearn.neighbors import KNeighborsClassifier

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

UNKNOWN_THRESHOLD = 0.9

# --- Cargar dataset y modelo ---
def load_data_model():
    dataset_path = "embeddings_data/faces_dataset.pkl"
    model_path = "modelos/knn_model.pkl"
    
    if os.path.exists(dataset_path):
        with open(dataset_path, "rb") as f:
            data = pickle.load(f)
        X_all = np.array(data["embeddings"])
        y_all = np.array(data["labels"])
    else:
        X_all, y_all = np.array([]), np.array([])

    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            knn_model = pickle.load(f)
    else:
        knn_model = None

    return X_all, y_all, knn_model

X_all, y_all, knn_model = load_data_model()

# --- Reconocimiento ---
@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    if knn_model is None:
        return {"error": "Modelo no cargado"}

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    frame = np.array(image)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = face_mesh.process(rgb_frame)

    if not results.multi_face_landmarks:
        return {"name": "No detectado", "confidence": 0}

    ih, iw, _ = frame.shape
    landmarks = results.multi_face_landmarks[0].landmark

    x_min = int(min([lm.x for lm in landmarks])*iw)
    x_max = int(max([lm.x for lm in landmarks])*iw)
    y_min = int(min([lm.y for lm in landmarks])*ih)
    y_max = int(max([lm.y for lm in landmarks])*ih)

    margin_x = int(0.1*(x_max-x_min))
    margin_y = int(0.1*(y_max-y_min))
    x_min = max(0, x_min - margin_x)
    x_max = min(iw, x_max + margin_x)
    y_min = max(0, y_min - margin_y)
    y_max = min(ih, y_max + margin_y)

    fc = frame[y_min:y_max, x_min:x_max]

    try:
        embedding = DeepFace.represent(fc, model_name="Facenet512", enforce_detection=False)
        r = np.array(embedding[0]["embedding"]).reshape(1, -1)
    except Exception as e:
        return {"error": f"No se pudo generar embedding: {str(e)}"}

    if hasattr(knn_model, "predict_proba"):
        probs = knn_model.predict_proba(r)[0]
        max_prob = np.max(probs)
        if max_prob < UNKNOWN_THRESHOLD:
            return {"name": "Cara no registrada", "confidence": float(max_prob)}
        else:
            pred_name = knn_model.classes_[np.argmax(probs)]
            return {"name": str(pred_name), "confidence": float(max_prob)*100}
    else:
        pred_name = knn_model.predict(r)[0]
        return {"name": str(pred_name), "confidence": float(max_prob)*100}

# --- Añadir nueva cara ---
# --- Añadir nueva cara con augmentación ---
@app.post("/add_face")
async def add_face(file: UploadFile = File(...), name: str = "", dataset_name: str = "faces_dataset.pkl", model_name: str = "knn_model.pkl"):
    global X_all, y_all, knn_model

    if not name:
        return {"error": "Debes especificar un nombre"}

    # Leer imagen
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    frame = np.array(image)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Detectar rostro
    results = face_mesh.process(rgb_frame)
    if not results.multi_face_landmarks:
        return {"error": "No se detectó rostro"}

    ih, iw, _ = frame.shape
    landmarks = results.multi_face_landmarks[0].landmark

    x_min = int(min([lm.x for lm in landmarks]) * iw)
    x_max = int(max([lm.x for lm in landmarks]) * iw)
    y_min = int(min([lm.y for lm in landmarks]) * ih)
    y_max = int(max([lm.y for lm in landmarks]) * ih)

    # recorte sin márgenes
    fc = frame[y_min:y_max, x_min:x_max]

    fc = cv2.resize(fc, (100, 100))

    # --- Data augmentation ---
    def augment_face(face):
        aug = [face]
        center = (face.shape[1]//2, face.shape[0]//2)
        for angle in [10, -10]:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aug.append(cv2.warpAffine(face, M, (face.shape[1], face.shape[0])))
        aug.append(cv2.flip(face, 1))
        aug.append(cv2.convertScaleAbs(face, alpha=1.0, beta=20))
        aug.append(cv2.convertScaleAbs(face, alpha=1.0, beta=-20))
        aug.append(cv2.GaussianBlur(face, (5,5), 0))
        return aug

    augmented_faces = augment_face(fc)

    embeddings_list = []
    for f in augmented_faces:
        try:
            emb = DeepFace.represent(f, model_name="Facenet512", enforce_detection=False)
            embeddings_list.append(np.array(emb[0]["embedding"]))
        except:
            continue

    if not embeddings_list:
        return {"error": "No se pudo generar embedding"}

    embeddings_array = np.vstack(embeddings_list)
    labels_array = np.array([name] * embeddings_array.shape[0])

    # --- Guardar dataset ---
    dataset_path = os.path.join("embeddings_data", dataset_name)
    os.makedirs("embeddings_data", exist_ok=True)

    # 🔹 Mostrar tamaño antes de actualizar
    if os.path.exists(dataset_path):
        with open(dataset_path, "rb") as f:
            data = pickle.load(f)
        print(f"[ANTES] Total embeddings en dataset: {data['embeddings'].shape[0]}")
        X_all = np.vstack([data["embeddings"], embeddings_array])
        y_all = np.append(data["labels"], labels_array)
    else:
        print("[ANTES] Dataset no existe, iniciando uno nuevo...")
        X_all = embeddings_array
        y_all = labels_array

    with open(dataset_path, "wb") as f:
        pickle.dump({"embeddings": X_all, "labels": y_all}, f)

    # 🔹 Mostrar tamaño después de actualizar
    print(f"[DESPUÉS] Total embeddings en dataset: {X_all.shape[0]}")

    # --- Reentrenar y guardar modelo ---
    knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')
    knn_model.fit(X_all, y_all)

    os.makedirs("modelos", exist_ok=True)
    model_path = os.path.join("modelos", model_name)
    with open(model_path, "wb") as f:
        pickle.dump(knn_model, f)

    return {
        "message": f"Cara de {name} añadida correctamente con {len(augmented_faces)} augmentaciones",
        "dataset_total": X_all.shape[0],
        "dataset_path": dataset_path,
        "model_path": model_path
    }