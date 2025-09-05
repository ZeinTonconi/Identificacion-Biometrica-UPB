import os
import time
import pickle
import numpy as np
import cv2
import mediapipe as mp
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from deepface import DeepFace
import subprocess

# Desactivar advertencias de oneDNN para reducir mensajes de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def list_available_cameras(max_test=10):
    """Detecta cámaras disponibles probando índices de 0 a max_test."""
    available_cameras = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras

def select_camera():
    """Muestra cámaras disponibles y permite al usuario seleccionar una."""
    cameras = list_available_cameras()
    if not cameras:
        print("No se encontraron cámaras disponibles.")
        exit()
    
    print("Cámaras disponibles (índices):", cameras)
    while True:
        try:
            camera_index = int(input("Ingrese el índice de la cámara a usar (por ejemplo, 0): "))
            if camera_index in cameras:
                return camera_index
            else:
                print(f"Índice inválido. Por favor, elija uno de: {cameras}")
        except ValueError:
            print("Entrada inválida. Ingrese un número entero.")

def preprocess_face(face_img, size=(100, 100)):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    resized = cv2.resize(gray, size)
    normalized = resized.astype("float32") / 255.0
    return normalized.flatten()

def iou(boxA, boxB):
    # Compute intersection over union between two bounding boxes (x, y, w, h)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]
    iou_value = inter_area / (boxA_area + boxB_area - inter_area) if (boxA_area + boxB_area - inter_area) > 0 else 0
    return iou_value

# Cargar CSV para mapear nombres a colegios
csv_path = 'csv_integrado/Datos_fotos.csv'
try:
    df = pd.read_csv(csv_path)
    name_to_college = dict(zip(df['Nombre'], df['Colegio']))
except Exception as e:
    print(f"Error al cargar el CSV: {e}")
    exit()

# Carpeta de datos unificados
data_dir = 'csv_integrado/dataUnificado'

# Verificar si existen archivos de embeddings, si no, ejecutar test.py
embeddings_files = [f for f in os.listdir(data_dir) if f.startswith('embeddings_') and f.endswith('.pkl')]
if not embeddings_files:
    print(f"No se encontraron archivos de embeddings en {data_dir}. Ejecutando test.py para generarlos...")
    try:
        subprocess.run(['python', 'test.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar test.py: {e}")
        exit()

# Cargar embeddings pregenerados
X_all, y_all = [], []
for file in os.listdir(data_dir):
    if file.startswith('embeddings_') and file.endswith('.pkl'):
        name = file.replace('embeddings_', '').replace('.pkl', '')
        try:
            embeddings = pickle.load(open(os.path.join(data_dir, file), 'rb'))
            X_all.extend(embeddings)
            y_all.extend([name] * len(embeddings))
            print(f"Cargado {file}: {len(embeddings)} embeddings")
        except Exception as e:
            print(f"Error al cargar {file}: {e}")
            continue

X_all = np.array(X_all)
y_all = np.array(y_all)

if len(X_all) == 0:
    print("No se encontraron embeddings válidos en dataUnificado. Ejecuta test.py primero.")
    exit()

# Entrenar KNN con embeddings
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_all, y_all)
print("Modelo KNN entrenado con embeddings.")

# Inicializar MediaPipe Face Mesh para detección y landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Seleccionar cámara
camera_index = select_camera()
camera = cv2.VideoCapture(camera_index)
if not camera.isOpened():
    print(f"No se pudo abrir la cámara con índice {camera_index}.")
    exit()

prev_bbox = None
iou_threshold = 0.5
confidence_threshold = 95.0  # Umbral de confianza ajustado a 75%
initial_delay = 3.0  # Retraso inicial de 3 segundos
consistency_time = 3.0  # Tiempo de consistencia de 3 segundos
welcome_interval = 3.0  # Intervalo para repetir mensajes de bienvenida
start_time = time.time()
last_name = None
name_start_time = None
last_welcome_time = None  # Para controlar el intervalo de mensajes de bienvenida

while True:
    ret, frame = camera.read()
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        face_coordinates = []

        if results.multi_face_landmarks:
            ih, iw, _ = frame.shape
            for landmarks in results.multi_face_landmarks:
                x_min = int(min([lm.x for lm in landmarks.landmark]) * iw)
                x_max = int(max([lm.x for lm in landmarks.landmark]) * iw)
                y_min = int(min([lm.y for lm in landmarks.landmark]) * ih)
                y_max = int(max([lm.y for lm in landmarks.landmark]) * ih)
                w = x_max - x_min
                h = y_max - y_min
                face_coordinates.append((x_min, y_min, w, h))
                break  # Solo tomar la primera cara detectada

        selected_bbox = None

        if prev_bbox is None:
            if len(face_coordinates) > 0:
                selected_bbox = face_coordinates[0]  # Tomar la primera cara
        else:
            if len(face_coordinates) > 0:
                ious = [iou(prev_bbox, fc) for fc in face_coordinates]
                max_iou_index = np.argmax(ious)
                if ious[max_iou_index] > iou_threshold:
                    selected_bbox = face_coordinates[max_iou_index]

        display_text = "Esperando..." if time.time() - start_time < initial_delay else "Cara no registrada"

        if selected_bbox is not None and time.time() - start_time >= initial_delay:
            a, b, w, h = selected_bbox
            # Crop la cara
            face = frame[b:b + h, a:a + w]
            if face.size == 0:
                continue
            # Resize a 100x100
            face_resized = cv2.resize(face, (100, 100))
            # Preprocesar a gray flatten
            face_processed = preprocess_face(face_resized)
            # Convertir a imagen para embedding
            face_img = (face_processed.reshape(100, 100) * 255).astype(np.uint8)
            face_img = np.stack([face_img] * 3, axis=-1)  # RGB
            # Obtener embedding
            try:
                embedding = DeepFace.represent(face_img, model_name='Facenet512', enforce_detection=False)[0]['embedding']
                embedding = np.array(embedding).reshape(1, -1)
                # Predecir
                text = knn.predict(embedding)
                proba = knn.predict_proba(embedding)[0]
                predicted_index = np.where(knn.classes_ == text[0])[0][0]
                conf = proba[predicted_index] * 100

                # Verificar consistencia del nombre
                if conf >= confidence_threshold:
                    if last_name == text[0]:
                        # Mismo nombre, verificar tiempo
                        if name_start_time is None:
                            name_start_time = time.time()
                        elif time.time() - name_start_time >= consistency_time:
                            # Ha pasado 3 segundos con el mismo nombre
                            display_text = f"{text[0]} ({conf:.2f}%)"
                            # Imprimir mensaje de bienvenida cada 3 segundos
                            if last_welcome_time is None or time.time() - last_welcome_time >= welcome_interval:
                                college = name_to_college.get(text[0], "Desconocido")
                                print(f"Bienvenido: {text[0]} del colegio {college}")
                                last_welcome_time = time.time()
                        else:
                            display_text = "Verificando..."
                    else:
                        # Nuevo nombre detectado, reiniciar temporizador
                        last_name = text[0]
                        name_start_time = time.time()
                        last_welcome_time = None
                        display_text = "Verificando..."
                else:
                    display_text = "Cara no registrada"
                    last_name = None
                    name_start_time = None
                    last_welcome_time = None
            except Exception as e:
                display_text = "Error en reconocimiento"
                print(f"Error generando embedding en tiempo real: {e}")
                last_name = None
                name_start_time = None
                last_welcome_time = None

            cv2.putText(frame, display_text, (a, b-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.rectangle(frame, (a, b), (a + w, b + h), (0, 0, 255), 2)
            prev_bbox = selected_bbox
        else:
            prev_bbox = None
            last_name = None
            name_start_time = None
            last_welcome_time = None

        cv2.imshow('livetime face recognition', frame)
        if cv2.waitKey(1) == 27:
            break
    else:
        print("error")
        break

cv2.destroyAllWindows()
camera.release()