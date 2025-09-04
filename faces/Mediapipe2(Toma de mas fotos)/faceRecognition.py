import cv2
from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np
import mediapipe as mp
import os
import glob

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

def preprocess_face(face_img, size=(100, 100)):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    resized = cv2.resize(gray, size)
    normalized = resized.astype("float32") / 255.0
    return normalized.flatten()

def list_available_cameras(max_test=10):
    print("Buscando cámaras disponibles...")
    available_cameras = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras

# List available cameras and prompt user to select one
available_cameras = list_available_cameras()
if not available_cameras:
    print("No se encontraron cámaras disponibles.")
    exit()

print("Cámaras disponibles (índices):", available_cameras)
camera_index = int(input("Ingresa el índice de la cámara a usar (por ejemplo, 0, 1, 2, ...): "))
if camera_index not in available_cameras:
    print("Índice de cámara no válido. Usando cámara predeterminada (0).")
    camera_index = 0

# Load all face and name files from the data directory
data_dir = 'data2'
faces = []
labels = []

face_files = glob.glob(os.path.join(data_dir, 'faces_*.pkl'))
name_files = glob.glob(os.path.join(data_dir, 'names_*.pkl'))

for face_file, name_file in zip(sorted(face_files), sorted(name_files)):
    with open(face_file, 'rb') as w:
        face_data = pickle.load(w)
        faces.append(face_data)
    with open(name_file, 'rb') as file:
        name_data = pickle.load(file)
        labels.extend(name_data)

# Convert to numpy arrays
if faces:
    faces = np.concatenate(faces, axis=0)
    labels = np.array(labels)
else:
    print("No se encontraron datos de rostros o nombres en la carpeta 'data3'.")
    exit()

camera = cv2.VideoCapture(camera_index)

print('Shape of Faces matrix --> ', faces.shape)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(faces, labels)

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

prev_bbox = None
iou_threshold = 0.5
confidence_threshold = 70.0  # 70% confidence threshold

while True:
    ret, frame = camera.read()
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        face_coordinates = []

        if results.detections:
            ih, iw, _ = frame.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = max(0, int(bbox.xmin * iw))
                y = max(0, int(bbox.ymin * ih))
                w = min(iw - x, int(bbox.width * iw))
                h = min(ih - y, int(bbox.height * ih))
                face_coordinates.append((x, y, w, h))

        selected_bbox = None

        if prev_bbox is None:
            # No previous face, select the largest (closest) if any
            if len(face_coordinates) > 0:
                areas = [w * h for (x, y, w, h) in face_coordinates]
                max_area_index = np.argmax(areas)
                selected_bbox = face_coordinates[max_area_index]
        else:
            # Track the previous face by finding the best IoU match
            if len(face_coordinates) > 0:
                ious = [iou(prev_bbox, fc) for fc in face_coordinates]
                max_iou_index = np.argmax(ious)
                if ious[max_iou_index] > iou_threshold:
                    selected_bbox = face_coordinates[max_iou_index]

        if selected_bbox is not None:
            a, b, w, h = selected_bbox
            # Process recognition
            fc = frame[b:b + h, a:a + w, :]
            r = preprocess_face(fc).reshape(1, -1)
            text = knn.predict(r)
            proba = knn.predict_proba(r)[0]
            predicted_index = np.where(knn.classes_ == text[0])[0][0]
            conf = proba[predicted_index] * 100
            # Display name only if confidence is 70% or higher
            display_text = f"{text[0]} ({conf:.2f}%)" if conf >= confidence_threshold else "Cara no registrada"
            cv2.putText(frame, display_text, (a, b-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.rectangle(frame, (a, b), (a + w, b + h), (0, 0, 255), 2)
            prev_bbox = selected_bbox  # Update previous bbox
        else:
            prev_bbox = None  # Lost track, reset

        cv2.imshow('livetime face recognition', frame)
        if cv2.waitKey(1) == 27:
            break
    else:
        print("error")
        break

cv2.destroyAllWindows()
camera.release()