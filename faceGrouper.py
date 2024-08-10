import os
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

# Create table for storing facial features and image paths
cursor.execute('''
CREATE TABLE IF NOT EXISTS Faces (
    id INTEGER PRIMARY KEY,
    features BLOB,
    image_paths TEXT
)
''')
conn.commit()

def get_face_features(image, face_rect):
    """Extract the 128D facial features from a face rectangle."""
    shape = predictor(image, face_rect)
    face_descriptor = face_rec_model.compute_face_descriptor(image, shape,num_jitters=10)
    return np.array(face_descriptor)

def match_face(features, tolerance=0.4):
    """Check if the face features match any in the database."""
    cursor.execute('SELECT id, features FROM Faces')
    for row in cursor.fetchall():
        db_id, db_features = row
        db_features = np.frombuffer(db_features, dtype=np.float64)
        distance = np.linalg.norm(features - db_features)
        if distance < tolerance:
            return True, db_id
    return False, -1

def store_face_in_db(features, image_path):
    """Store the new face features and image path in the database."""
    features_blob = features.tobytes()
    cursor.execute('INSERT INTO Faces (features, image_paths) VALUES (?, ?)', (features_blob, image_path))
    conn.commit()

def update_image_paths(db_id, image_path):
    """Update the image paths for an existing face in the database."""
    cursor.execute('SELECT image_paths FROM Faces WHERE id = ?', (db_id,))
    existing_paths = cursor.fetchone()[0]
    new_paths = existing_paths + "," + image_path
    cursor.execute('UPDATE Faces SET image_paths = ? WHERE id = ?', (new_paths, db_id))
    conn.commit()

def process_image(image_path):
    """Process a single image to detect and identify faces."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        print(f"No faces found in {image_path}")
        return 0
    
    new_faces = 0
    for face in faces:
        # Pass the original color image to compute_face_descriptor
        features = get_face_features(image, face)
        match, db_id = match_face(features)
        
        if match:
            print(f"Match found in {image_path}")
            update_image_paths(db_id, image_path)
        else:
            print(f"New face found in {image_path}")
            store_face_in_db(features, image_path)
            new_faces += 1
    
    return new_faces

def process_directory(directory):
    """Process all images in the directory."""
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more extensions as needed
            image_path = os.path.join(directory, filename)
            process_image(image_path)

    print("Processing complete.")

# Path to the directory containing the images
image_directory = 'Testing'

process_directory(image_directory)

# Close the database connection when done
conn.close()
