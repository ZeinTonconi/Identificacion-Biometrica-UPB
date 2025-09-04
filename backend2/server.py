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

# 🔹 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 🔹 CARGAR DATOS Y ENTRENAR KNN ---
data_dir = "../csv_integrado/dataUnificado"
faces_list = []
labels_list = []

face_files = sorted(glob.glob(os.path.join(data_dir, "faces_*.pkl")))
name_files = sorted(glob.glob(os.path.join(data_dir, "names_*.pkl")))

for face_file, name_file in zip(face_files, name_files):
    with open(face_file, "rb") as f:
        faces = pickle.load(f)
        faces_list.append(faces)
    with open(name_file, "rb") as f:
        names = pickle.load(f)
        labels_list.extend(names)

if faces_list:
    X_train = np.concatenate(faces_list, axis=0)
    y_train = np.array(labels_list)
    print("✅ Datos cargados:", X_train.shape, "caras")
else:
    raise Exception("❌ No se encontraron datos en la carpeta 'data2'")

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

# --- 🔹 INITIALIZAR MEDIA PIPE FACE MESH ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

confidence_threshold = 70.0  # umbral mínimo de confianza

# --- 🔹 ENDPOINT DE RECONOCIMIENTO ---
@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    frame = np.array(image)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        ih, iw, _ = frame.shape
        landmarks = results.multi_face_landmarks[0].landmark

        # Bounding box mínimo de todos los landmarks
        x_min = int(min([lm.x for lm in landmarks]) * iw)
        x_max = int(max([lm.x for lm in landmarks]) * iw)
        y_min = int(min([lm.y for lm in landmarks]) * ih)
        y_max = int(max([lm.y for lm in landmarks]) * ih)

        # 🔹 Expandir bounding box para incluir orejas (10% extra)
        margin_x = int(0.1 * (x_max - x_min))
        margin_y = int(0.1 * (y_max - y_min))
        x_min = max(0, x_min - margin_x)
        x_max = min(iw, x_max + margin_x)
        y_min = max(0, y_min - margin_y)
        y_max = min(ih, y_max + margin_y)

        # Crop del rostro
        fc = frame[y_min:y_max, x_min:x_max]

        # Guardar imagen recortada
        os.makedirs("temp_crops", exist_ok=True)
        cv2.imwrite("temp_crops/face_crop.jpg", cv2.cvtColor(fc, cv2.COLOR_RGB2BGR))

        # Guardar imagen original con rectángulo y etiqueta
        frame_with_box = frame.copy()
        cv2.rectangle(frame_with_box, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Preprocesamiento para KNN
        fc_gray = cv2.cvtColor(fc, cv2.COLOR_BGR2GRAY)
        r = cv2.resize(fc_gray, (100, 100)).flatten().reshape(1, -1)

        # Predicción KNN
        text = knn.predict(r)[0]
        proba = knn.predict_proba(r)[0]
        predicted_index = np.where(knn.classes_ == text)[0][0]
        conf = proba[predicted_index] * 100

        label = f"{text} ({conf:.1f}%)" if conf >= confidence_threshold else f"Cara no registrada ({conf:.1f}%)"
        cv2.putText(frame_with_box, label, (x_min, max(0, y_min-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imwrite("temp_crops/original_with_box.jpg", cv2.cvtColor(frame_with_box, cv2.COLOR_RGB2BGR))

        return {"name": text if conf >= confidence_threshold else "Cara no registrada", "confidence": conf}

    return {"name": "No detectado", "confidence": 0}

