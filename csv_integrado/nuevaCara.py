import os
import numpy as np
import cv2
import pickle
import mediapipe as mp
import pandas as pd

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

camera = cv2.VideoCapture(camera_index)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

nombre = input('Ingresa tu nombre y apellido (ej: Abigail Ballesteros) --> ')
colegio = input('Ingresa el nombre del colegio/institución (ej: San Ignacio) --> ')
ret = True

captured = False
tracked_bbox = None
iou_threshold = 0.5

# Directorios y paths
image_dir = 'csv_integrado/fotos de chicos 100 M 2025-20250903T165147Z-1-001/fotos de chicos 100 M 2025'
csv_path = 'csv_integrado/Datos_fotos.csv'

# Asegurar que el directorio de imágenes existe
os.makedirs(image_dir, exist_ok=True)

# Asegurar que el directorio del CSV existe
os.makedirs(os.path.dirname(csv_path), exist_ok=True)

while ret:
    ret, frame = camera.read()
    if ret:
        # Crear una copia del frame para mostrar con rectángulo y texto
        display_frame = frame.copy()
        
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

        selected_bbox = None

        if tracked_bbox is None:
            # No tracked face, select the largest (closest) if any
            if len(face_coordinates) > 0:
                areas = [w * h for (x, y, w, h) in face_coordinates]
                max_area_index = np.argmax(areas)
                selected_bbox = face_coordinates[max_area_index]
                tracked_bbox = selected_bbox
        else:
            # Track the first detected face using IoU
            if len(face_coordinates) > 0:
                ious = [iou(tracked_bbox, fc) for fc in face_coordinates]
                max_iou_index = np.argmax(ious)
                if ious[max_iou_index] > iou_threshold:
                    selected_bbox = face_coordinates[max_iou_index]
                    tracked_bbox = selected_bbox
                else:
                    tracked_bbox = None  # Lost track, reset

        # Dibujar rectángulo y texto solo en la copia para mostrar
        if selected_bbox is not None:
            a, b, w, h = selected_bbox
            cv2.rectangle(display_frame, (a, b), (a + w, b + h), (255, 0, 0), 2)

        # Mostrar instrucción solo en la copia para mostrar
        cv2.putText(display_frame, "Presiona 'c' para capturar (ESC para salir)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('frames', display_frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC to exit
            break
        elif key == ord('c') and selected_bbox is not None and not captured:
            # Construir el nombre del archivo
            filename = f"{nombre} - {colegio}.jpeg"
            img_path = os.path.join(image_dir, filename)

            # Guardar la imagen original (sin rectángulo ni texto)
            cv2.imwrite(img_path, frame)
            print(f"Foto guardada en: {img_path}")

            # Actualizar el CSV
            new_row = {'Nombre': nombre, 'Colegio': colegio, 'Nombre del Archivo': filename}
            try:
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                else:
                    # Crear un DataFrame vacío con las columnas adecuadas si el archivo no existe
                    df = pd.DataFrame(columns=['Nombre', 'Colegio', 'Nombre del Archivo'])
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                df.to_csv(csv_path, index=False)
                print(f"Entrada agregada al CSV: {csv_path}")
            except Exception as e:
                print(f"Error al actualizar el CSV: {e}")
                break

            captured = True
            print("Foto capturada y registrada! Ahora ejecuta nuevoCodigo.py para procesar los .pkl.")
            break

    else:
        print('error')
        break

cv2.destroyAllWindows()
camera.release()

if not captured:
    print("No se capturó ninguna foto.")