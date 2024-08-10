import face_recognition
import os
import sqlite3
import numpy as np

# Create a connection to the SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('faces.db')
c = conn.cursor()

# Delete tables if they exist
c.execute('''
DROP TABLE IF EXISTS unique_faces
''')
c.execute('''
DROP TABLE IF EXISTS face_locations
''')

# Create tables for storing face encodings and file locations
c.execute('''
CREATE TABLE IF NOT EXISTS unique_faces (
    id INTEGER PRIMARY KEY,
    encoding BLOB,
    primary_file_location TEXT
)
''')
c.execute('''
CREATE TABLE IF NOT EXISTS face_locations (
    face_id INTEGER,
    file_location TEXT,
    FOREIGN KEY(face_id) REFERENCES unique_faces(id)
)
''')

conn.commit()

def process_images(image_paths):
    for image_path in image_paths:
        # Load the image
        image = face_recognition.load_image_file(image_path)

        # Detect faces and extract embeddings using the 'cnn' model for better accuracy
        face_locations = face_recognition.face_locations(image, model="cnn")
        face_encodings = face_recognition.face_encodings(image, face_locations)

        # Print the number of faces detected
        num_faces = len(face_locations)
        print(f"{num_faces} face(s) detected in {image_path}")

        # Store face encodings and locations in the database
        for encoding in face_encodings:
            store_face(encoding, image_path)

def store_face(encoding, file_location):
    # Check if the face is unique and store it
    is_unique, face_id = is_unique_face(encoding)
    
    if is_unique:
        encoding_blob = sqlite3.Binary(encoding.tobytes())  # convert encoding to binary
        c.execute("INSERT INTO unique_faces (encoding, primary_file_location) VALUES (?, ?)",
                  (encoding_blob, file_location))
        face_id = c.lastrowid
    
    c.execute("INSERT INTO face_locations (face_id, file_location) VALUES (?, ?)", (face_id, file_location))
    conn.commit()

def is_unique_face(new_encoding, threshold=0.4):
    c.execute("SELECT id, encoding FROM unique_faces")
    rows = c.fetchall()
    
    for row in rows:
        face_id, encoding_blob = row
        known_encoding = np.frombuffer(encoding_blob, dtype=np.float64)
        distance = face_recognition.face_distance([known_encoding], new_encoding)[0]
        
        print(f"Face ID: {face_id}")
        print(f"Distance: {distance}")
        print(f"Threshold: {threshold}")
        
        
        if distance < threshold:
            return False, face_id
    return True, None

# Example usage with multiple image paths
image_directory = "Testing"  # Replace with your directory containing images
image_paths = [os.path.join(image_directory, filename) for filename in os.listdir(image_directory) if filename.endswith(('.jpg', '.png'))]

process_images(image_paths)

# Close the database connection
conn.close()
