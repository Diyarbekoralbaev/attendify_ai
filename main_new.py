# main_new.py
import os
import threading
import time
from datetime import datetime
import cv2
import numpy as np
from dotenv import load_dotenv
from insightface.app import FaceAnalysis
from pymongo import MongoClient
from bson.binary import Binary
from funcs import compute_sim, extract_date_from_filename, get_faces_data, setup_logger

load_dotenv()


class Config:
    CHECK_NEW_CLIENT = 0.5  # Similarity threshold for clients
    EMPLOYEE_SIMILARITY_THRESHOLD = 0.5  # Similarity threshold for employees
    MIN_DETECTION_CONFIDENCE = 0.6 # Minimum detection confidence for faces
    DET_SCORE_THRESH = 0.65
    POSE_THRESHOLD = 40
    logger = setup_logger('MainRunner', 'logs/main.log')
    DIMENSIONS = 512
    DET_SIZE = (640, 640)


class Database:
    def __init__(self):
        self.client = MongoClient(os.getenv('MONGODB_LOCAL'))
        self.db = self.client.empl_time
        self.employees = self.db.employees
        self.clients = self.db.clients
        self.attendance = self.db.attendance

    def find_matching_client(self, embedding):
        """Find existing client with closest matching face embedding"""
        all_clients = self.clients.find({"embedding": {"$exists": True}})
        best_match = None
        highest_similarity = 0

        for client in all_clients:
            client_embedding = np.array(client['embedding'])
            similarity = compute_sim(client_embedding, embedding)
            if similarity is not None and similarity > Config.CHECK_NEW_CLIENT and similarity > highest_similarity:
                highest_similarity = similarity
                best_match = client

        return best_match, highest_similarity

    def save_attendance(self, person_id, camera_id, image_data, timestamp, score):
        attendance_data = {
            "user_id": person_id,
            "device_id": camera_id,
            "image": Binary(image_data),
            "timestamp": timestamp,
            "score": float(score),  # Convert to Python float
            "created_at": datetime.now()
        }
        self.attendance.insert_one(attendance_data)
        Config.logger.info(f"Attendance saved for person {person_id}")

    def save_employee(self, person_id, image_data, embedding):
        # Remove existing employee data if any
        self.employees.delete_many({"person_id": person_id})

        employee_data = {
            "person_id": person_id,
            "image": Binary(image_data),
            "embedding": embedding.tolist(),
            "created_at": datetime.now()
        }
        self.employees.insert_one(employee_data)
        Config.logger.info(f"Employee saved with ID: {person_id}")

    def update_client(self, client_id, image_data, embedding, face_data):
        """Update existing client with new visit data"""
        update_data = {
            "$set": {
                "last_visit": datetime.now(),
                "image": Binary(image_data),
                "embedding": embedding.tolist(),
                "gender": int(face_data.gender),
                "age": int(face_data.age),
            },
            "$inc": {
                "visit_count": 1
            },
            "$push": {
                "visit_history": {
                    "timestamp": datetime.now(),
                    "image": Binary(image_data)
                }
            }
        }
        self.clients.update_one({"person_id": client_id}, update_data)
        Config.logger.info(f"Client {client_id} visit count updated")

    def save_client(self, person_id, image_data, embedding, face_data):
        """Save new client with initial visit data"""
        client_data = {
            "person_id": person_id,
            "image": Binary(image_data),
            "embedding": embedding.tolist(),
            "gender": int(face_data.gender),
            "age": int(face_data.age),
            "first_visit": datetime.now(),
            "last_visit": datetime.now(),
            "visit_count": 1,
            "visit_history": [{
                "timestamp": datetime.now(),
                "image": Binary(image_data)
            }],
            "created_at": datetime.now()
        }
        self.clients.insert_one(client_data)
        Config.logger.info(f"New client saved with ID: {person_id}")


class FaceProcessor:
    def __init__(self):
        # Initialize FaceAnalysis with desired models
        self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def process_image(self, image):
        faces = self.app.get(image)
        Config.logger.debug(f"Total faces detected before filtering: {len(faces)}")
        # Filter faces based on detection confidence
        faces = [face for face in faces if face.det_score >= Config.MIN_DETECTION_CONFIDENCE]
        Config.logger.debug(f"Faces after filtering: {len(faces)}")
        return faces


class MainRunner:
    def __init__(self, images_folder):
        self.images_folder = images_folder
        self.db = Database()
        self.face_processor = FaceProcessor()
        self.lock = threading.Lock()

    def process_faces(self, face_data, image_data, camera_id, date):
        # Fetch all employees with embeddings
        all_employees = list(self.db.employees.find({
            "embedding": {"$exists": True}
        }))

        best_match_employee = None
        highest_similarity = 0

        # Compare face embedding with each employee embedding
        for employee in all_employees:
            employee_embedding = np.array(employee['embedding'])
            similarity = compute_sim(employee_embedding, face_data.embedding)
            if similarity is not None and similarity > highest_similarity:
                highest_similarity = similarity
                best_match_employee = employee

        # Check if similarity exceeds the threshold
        if highest_similarity >= Config.EMPLOYEE_SIMILARITY_THRESHOLD:
            # Employee recognized
            self.db.save_attendance(
                best_match_employee['person_id'],
                camera_id,
                image_data,
                date.strftime("%Y-%m-%d %H:%M:%S"),
                face_data.det_score
            )
            Config.logger.info(f"Attendance saved for employee {best_match_employee['person_id']} with similarity {highest_similarity}")
            return

        # Check against clients
        matching_client, similarity = self.db.find_matching_client(face_data.embedding)

        if matching_client:
            # Update existing client's visit data
            self.db.update_client(
                matching_client['person_id'],
                image_data,
                face_data.embedding,
                face_data
            )
            Config.logger.info(f"Updated existing client {matching_client['person_id']} with similarity {similarity}")
        else:
            # Add as new client
            new_id = self.db.clients.count_documents({}) + 1
            self.db.save_client(
                new_id,
                image_data,
                face_data.embedding,
                face_data
            )
            Config.logger.info(f"Saved new client with ID {new_id}")


    def process_image_file(self, file_path, camera_id):
        try:
            image = cv2.imread(file_path)
            if image is None:
                Config.logger.error(f"Failed to read image from {file_path}")
                return

            # Convert from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            Config.logger.debug(f"Image shape: {image.shape}")

            # Resize image to standard size
            image = cv2.resize(image, (640, 480))

            faces = self.face_processor.process_image(image)
            Config.logger.info(f"Processing file: {file_path}, Faces detected: {len(faces)}")
            if not faces:
                Config.logger.error(f"No faces found in the image: {file_path}")
                return

            face_data = get_faces_data(faces, min_confidence=Config.MIN_DETECTION_CONFIDENCE)
            if not face_data:
                Config.logger.error(f"Could not extract face data from image: {file_path}")
                return

            date = extract_date_from_filename(os.path.basename(file_path))
            if not date:
                Config.logger.error(f"Could not extract date from filename: {file_path}")
                return

            # Read image data as bytes
            with open(file_path, 'rb') as f:
                image_data = f.read()

            self.process_faces(face_data, image_data, camera_id, date)

        except Exception as e:
            Config.logger.error(f"Error processing image {file_path}: {e}")
        finally:
            # Clean up the processed file
            if os.path.exists(file_path):
                os.remove(file_path)
            # Remove corresponding BACKGROUND file if it exists
            bg_file = file_path.replace('SNAP', 'BACKGROUND')
            if os.path.exists(bg_file):
                os.remove(bg_file)

    def run(self):
        Config.logger.info(f"Checking directory: {self.images_folder}")
        while True:

            try:
                # Process only test_camera directory
                camera_dir = os.path.join(self.images_folder, 'test_camera')

                if not os.path.exists(camera_dir):
                    time.sleep(1)
                    continue

                for file in os.listdir(camera_dir):
                    if file.endswith('SNAP.jpg'):
                        file_path = os.path.join(camera_dir, file)
                        self.process_image_file(file_path, 1)  # Using camera_id = 1

            except Exception as e:
                Config.logger.error(f'Exception in main run: {e}')
            time.sleep(1)

    def add_employee(self, image_path, person_id):
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Failed to read image")

            # Convert from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize image to standard size
            image = cv2.resize(image, (640, 480))

            faces = self.face_processor.process_image(image)
            if not faces:
                raise ValueError("No face detected in the image")

            face_data = get_faces_data(faces, min_confidence=Config.MIN_DETECTION_CONFIDENCE)
            if not face_data:
                raise ValueError("Could not process face data")

            with open(image_path, 'rb') as f:
                image_data = f.read()

            self.db.save_employee(person_id, image_data, face_data.embedding)
            return True
        except Exception as e:
            Config.logger.error(f"Error adding employee: {e}")
            return False


if __name__ == '__main__':
    runner = MainRunner(os.getenv('IMAGES_FOLDER'))
    runner.run()
