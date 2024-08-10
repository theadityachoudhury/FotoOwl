import face_recognition
import sqlite3
import numpy as np

# Connect to the SQLite database
conn = sqlite3.connect('faces.db')
c = conn.cursor()

def find_similar_faces(new_image_path, threshold=0.7):
    # Load the new image and detect faces
    new_image = face_recognition.load_image_file(new_image_path)
    new_face_encodings = face_recognition.face_encodings(new_image)
    
    if not new_face_encodings:
        print("No faces found in the provided image.")
        return []

    matched_locations = []

    # Iterate over each face found in the new image
    for new_encoding in new_face_encodings:
        # Compare with each encoding in the database
        c.execute("SELECT id, encoding FROM unique_faces")
        rows = c.fetchall()
        for row in rows:
            face_id, encoding_blob = row
            known_encoding = np.frombuffer(encoding_blob, dtype=np.float64)  # Convert binary to numpy array
            
            # Compare the new face encoding with the known encoding
            distance = face_recognition.face_distance([known_encoding], new_encoding)
            if distance < threshold:
                print(f"Face match found with distance: {distance} {face_id}")
                # If a match is found, retrieve all file locations for that face_id
                c.execute("SELECT file_location FROM face_locations WHERE face_id=?", (face_id,))
                locations = c.fetchall()
                matched_locations.extend([location[0] for location in locations])

    # Remove duplicates from matched locations
    matched_locations = list(set(matched_locations))
    return matched_locations

# Example usage
new_image_path = "Testing/IMG-20240726-WA0019.jpg"  # Replace with the path to your new image

# Find all matching face locations
matching_files = find_similar_faces(new_image_path)

if matching_files:
    print("Matching faces found in the following files:")
    for file in matching_files:
        print(file)
else:
    print("No matching faces found.")

# Close the database connection
conn.close()
