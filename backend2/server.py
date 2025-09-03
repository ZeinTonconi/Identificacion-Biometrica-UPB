from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import pickle
import mediapipe as mp
from sklearn.neighbors import KNeighborsClassifier
import io
from PIL import Image
import os
import glob

app = FastAPI()

# 🔹 CORS (permitir llamadas desde React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 🔹 ENTRENAMIENTO AL INICIAR SERVIDOR ---
data_dir = "../data2"
faces = []
labels = []

face_files = glob.glob(os.path.join(data_dir, "faces_*.pkl"))
name_files = glob.glob(os.path.join(data_dir, "names_*.pkl"))

for face_file, name_file in zip(sorted(face_files), sorted(name_files)):
    with open(face_file, "rb") as w:
        face_data = pickle.load(w)
        faces.append(face_data)
    with open(name_file, "rb") as f:
        name_data = pickle.load(f)
        labels.extend(name_data)

if faces:
    faces = np.concatenate(faces, axis=0)
    labels = np.array(labels)
    print("✅ Datos cargados:", faces.shape, "caras")
else:
    raise Exception("❌ No se encontraron datos en la carpeta 'data2'")

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(faces, labels)

# --- 🔹 Mediapipe ---
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

confidence_threshold = 60.0  # umbral mínimo

# --- 🔹 ENDPOINT DE RECONOCIMIENTO ---
@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    # Leer imagen desde el frontend
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    frame = np.array(image)

    # Convertir a RGB para Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    results = face_detection.process(frame)
    if results.detections:
        ih, iw, _ = frame.shape
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x = max(0, int(bbox.xmin * iw))
            y = max(0, int(bbox.ymin * ih))
            w = min(iw - x, int(bbox.width * iw))
            h = min(ih - y, int(bbox.height * ih))

            # Extraer rostro
            fc = frame[y:y+h, x:x+w, :]
            if fc.size == 0:
                continue
            r = cv2.resize(fc, (50, 50)).flatten().reshape(1, -1)

            # Predicción con KNN
            text = knn.predict(r)[0]
            proba = knn.predict_proba(r)[0]
            predicted_index = np.where(knn.classes_ == text)[0][0]
            conf = proba[predicted_index] * 100

            if conf >= confidence_threshold:
                return {"name": text, "confidence": conf}
            else:
                return {"name": "Cara no registrada", "confidence": conf}

    return {"name": "No detectado", "confidence": 0}
