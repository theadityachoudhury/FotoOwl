import cv2
import dlib
import numpy as np
import sqlite3

# Load the detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Initialize SQLite database
conn = sqlite3.connect('facial_features.db')
cursor = conn.cursor()

def get_face_features(image, face_rect):
    """Extract the 128D facial features from a face rectangle."""
    shape = predictor(image, face_rect)
    face_descriptor = face_rec_model.compute_face_descriptor(image, shape)
    return np.array(face_descriptor)

def find_matching_images(features, tolerance=0.53):
    """Find all images in the database that match the given facial features."""
    cursor.execute('SELECT id, features, image_paths FROM Faces')
    matching_images = []
    
    for row in cursor.fetchall():
        db_id, db_features, image_paths = row
        db_features = np.frombuffer(db_features, dtype=np.float64)
        distance = np.linalg.norm(features - db_features)
        if distance < tolerance:
            matching_images.extend(image_paths.split(','))
    
    return matching_images

def process_input_image(image_path):
    """Process the input image to find all matching images in the database."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        print(f"No faces found in {image_path}")
        return []
    
    all_matching_images = set()
    for face in faces:
        features = get_face_features(image, face)
        matching_images = find_matching_images(features)
        all_matching_images.update(matching_images)
    
    return list(all_matching_images)

# Path to the input image
input_image_path = 'Testing/tommy-testing.jpeg'

matching_images = process_input_image(input_image_path)

if matching_images:
    print(f"Matching images found for {input_image_path}:")
    for img_path in matching_images:
        print(img_path)
else:
    print(f"No matching images found for {input_image_path}")

# Close the database connection when done
conn.close()
