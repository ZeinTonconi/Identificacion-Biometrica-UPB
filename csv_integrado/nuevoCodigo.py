import os
import numpy as np
import cv2
import pickle
import mediapipe as mp
import pandas as pd

def preprocess_face(face_img, size=(100, 100)):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    resized = cv2.resize(gray, size)
    normalized = resized.astype("float32") / 255.0
    return normalized.flatten()

def augment_face(face):
    """Return list of augmented faces (preprocessed)."""
    augmentations = []
    # Original
    augmentations.append(preprocess_face(face))
    # Rotate +10 degrees
    center = (face.shape[1] // 2, face.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, 10, 1.0)
    rot1 = cv2.warpAffine(face, M, (face.shape[1], face.shape[0]))
    augmentations.append(preprocess_face(rot1))
    # Rotate -10 degrees
    M = cv2.getRotationMatrix2D(center, -10, 1.0)
    rot2 = cv2.warpAffine(face, M, (face.shape[1], face.shape[0]))
    augmentations.append(preprocess_face(rot2))
    # Horizontal flip
    flipped = cv2.flip(face, 1)
    augmentations.append(preprocess_face(flipped))
    # Increase brightness
    bright = cv2.convertScaleAbs(face, alpha=1.0, beta=20)
    augmentations.append(preprocess_face(bright))
    # Decrease brightness
    dark = cv2.convertScaleAbs(face, alpha=1.0, beta=-20)
    augmentations.append(preprocess_face(dark))
    # Blurred
    blurred = cv2.GaussianBlur(face, (5, 5), 0)
    augmentations.append(preprocess_face(blurred))
    return augmentations

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# Load CSV
csv_path = 'csv_integrado/Datos_fotos.csv'
df = pd.read_csv(csv_path)

# Image directory
image_dir = 'csv_integrado/fotos de chicos 100 M 2025-20250903T165147Z-1-001/fotos de chicos 100 M 2025'

# Output directory
output_dir = 'csv_integrado/dataUnificado'
os.makedirs(output_dir, exist_ok=True)

# Group by 'Nombre'
for name, group in df.groupby('Nombre'):
    # Verificar si los archivos .pkl ya existen
    face_filename = os.path.join(output_dir, f'faces_{name}.pkl')
    name_filename = os.path.join(output_dir, f'names_{name}.pkl')
    
    if os.path.exists(face_filename) and os.path.exists(name_filename):
        print(f"Archivos .pkl ya existen para {name}. Omitiendo procesamiento.")
        continue

    face_data_list = []
    names_list = []

    for _, row in group.iterrows():
        filename = row['Nombre del Archivo']
        img_path = os.path.join(image_dir, filename)

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Could not load image {filename}. Skipping.")
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            ih, iw, _ = frame.shape
            # Get bounding box from landmarks
            x_min = int(min([lm.x for lm in landmarks]) * iw)
            x_max = int(max([lm.x for lm in landmarks]) * iw)
            y_min = int(min([lm.y for lm in landmarks]) * ih)
            y_max = int(max([lm.y for lm in landmarks]) * ih)

            # Crop the face
            face = frame[y_min:y_max, x_min:x_max]

            # Resize to 100x100
            face = cv2.resize(face, (100, 100))

            # Augment and preprocess
            augmented = augment_face(face)
            face_data_list.extend(augmented)
            names_list.extend([name] * len(augmented))

            print(f"Processed image {filename} for {name} with {len(augmented)} augmentations.")
        else:
            print(f"Warning: No face detected in {filename}. Skipping.")

    if face_data_list:
        face_data = np.asarray(face_data_list)
        print(f"Shape of features for {name}: {face_data.shape}")

        with open(face_filename, 'wb') as w:
            pickle.dump(face_data, w)

        with open(name_filename, 'wb') as file:
            pickle.dump(names_list, file)

        print(f"Data saved for {name}.")
    else:
        print(f"No valid data processed for {name}.")

print("Processing complete. New data unified and saved in 'csv_integrado/dataUnificado/' folder.")