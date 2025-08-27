import os
import numpy as np
import cv2
import pickle
import mediapipe as mp

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

def augment_face(face):
    """Return list of augmented faces (preprocessed)."""
    augmentations = []
    # Original
    augmentations.append(preprocess_face(face))
    # Rotate +10 degrees (simulate head tilt right)
    center = (face.shape[1] // 2, face.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, 10, 1.0)
    rot1 = cv2.warpAffine(face, M, (face.shape[1], face.shape[0]))
    augmentations.append(preprocess_face(rot1))
    # Rotate -10 degrees (simulate head tilt left)
    M = cv2.getRotationMatrix2D(center, -10, 1.0)
    rot2 = cv2.warpAffine(face, M, (face.shape[1], face.shape[0]))
    augmentations.append(preprocess_face(rot2))
    # Horizontal flip (simulate mirror image for facial symmetry)
    flipped = cv2.flip(face, 1)
    augmentations.append(preprocess_face(flipped))
    # Increase brightness (handle varying lighting conditions)
    bright = cv2.convertScaleAbs(face, alpha=1.0, beta=20)
    augmentations.append(preprocess_face(bright))
    # Decrease brightness (handle darker conditions)
    dark = cv2.convertScaleAbs(face, alpha=1.0, beta=-20)
    augmentations.append(preprocess_face(dark))
    # Blurred (handle slight motion blur)
    blurred = cv2.GaussianBlur(face, (5, 5), 0)
    augmentations.append(preprocess_face(blurred))
    return augmentations

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

face_data = []
names = []
camera = cv2.VideoCapture(camera_index)
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

name = input('Ingresa tu nombre--> ')
ret = True

captured_count = 0
max_captures = 10  # Ajustable: número máximo de capturas raw
tracked_bbox = None
iou_threshold = 0.5

print(f"Presiona 'c' para capturar hasta {max_captures} veces (ESC para salir).")

while ret:
    ret, frame = camera.read()
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        face_coordinates = []

        if results.detections:
            ih, iw, _ = frame.shape
            for detection in results.detections:
                bbox = detection.location_data3.relative_bounding_box
                x = max(0, int(bbox.xmin * iw))
                y = max(0, int(bbox.ymin * ih))
                w = min(iw - x, int(bbox.width * iw))
                h = min(ih - y, int(bbox.height * ih))
                face_coordinates.append((x, y, w, h))

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

        # Draw rectangle only for the selected/tracked face
        if selected_bbox is not None:
            a, b, w, h = selected_bbox
            cv2.rectangle(frame, (a, b), (a + w, b + h), (255, 0, 0), 2)

        # Display instruction and capture count
        cv2.putText(frame, f"Presiona 'c' para capturar ({captured_count}/{max_captures}) (ESC para salir)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('frames', frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC to exit
            break
        elif key == ord('c') and selected_bbox is not None and captured_count < max_captures:
            # Capture the selected face
            a, b, w, h = selected_bbox
            face = frame[b:b+h, a:a+w]
            
            # Augment and preprocess
            augmented = augment_face(face)
            face_data.extend(augmented)
            names.extend([name] * len(augmented))
            
            captured_count += 1
            print(f"Captura {captured_count}/{max_captures} guardada con {len(augmented)} augmentaciones.")
    else:
        print('error')
        break

cv2.destroyAllWindows()
camera.release()

if len(face_data) > 0:
    face_data = np.asarray(face_data)
    print("Shape of saved features:", face_data.shape)
    
    # Create directory if it doesn't exist
    os.makedirs('data3', exist_ok=True)
    
    # Save face data with unique filename based on name
    face_filename = os.path.join('data3', f'faces_{name}.pkl')
    with open(face_filename, 'wb') as w:
        pickle.dump(face_data, w)
    
    # Save name data with corresponding filename
    name_filename = os.path.join('data3', f'names_{name}.pkl')
    with open(name_filename, 'wb') as file:
        pickle.dump(names, file)
else:
    print("No se capturo ninguna foto.")