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
import requests
from funcs import compute_sim, extract_date_from_filename, get_faces_data, setup_logger

load_dotenv()


class Config:
    CHECK_NEW_CLIENT = 0.5  # Similarity threshold for clients
    EMPLOYEE_SIMILARITY_THRESHOLD = 0.5  # Similarity threshold for employees
    MIN_DETECTION_CONFIDENCE = 0.6  # Minimum detection confidence for faces
    DET_SCORE_THRESH = 0.65
    POSE_THRESHOLD = 40
    logger = setup_logger('MainRunner', 'logs/main.log')
    DIMENSIONS = 512
    DET_SIZE = (640, 640)
    API_BASE_URL = os.getenv('API_BASE_URL', 'http://10.30.10.136:8000')
    HEADERS = {
        'Authorization': f'Bearer {os.getenv("API_TOKEN")}'  # If authentication is required
    }


def send_report(endpoint, data, files=None):
    url = f"{Config.API_BASE_URL}{endpoint}"
    try:
        response = requests.post(url, data=data, files=files, headers=Config.HEADERS)
        response.raise_for_status()
        Config.logger.info(f"Successfully sent report to {endpoint}")
    except requests.RequestException as e:
        Config.logger.error(f"Failed to send report to {endpoint}: {e}")


class Database:
    def __init__(self):
        self.client = MongoClient(os.getenv('MONGODB_LOCAL'))
        self.db = self.client.empl_time_fastapi
        self.employees = self.db.employees
        self.clients = self.db.clients
        self.attendance = self.db.attendance

    def find_matching_employee(self, embedding):
        """Find existing employee with closest matching face embedding"""
        all_employees = list(self.employees.find({"embedding": {"$exists": True}}))
        best_match = None
        highest_similarity = 0

        for employee in all_employees:
            employee_embedding = np.array(employee['embedding'])
            similarity = compute_sim(employee_embedding, embedding)
            if similarity is not None and similarity > Config.EMPLOYEE_SIMILARITY_THRESHOLD and similarity > highest_similarity:
                highest_similarity = similarity
                best_match = employee

        return best_match, highest_similarity

    def find_matching_client(self, embedding):
        """Find existing client with closest matching face embedding"""
        all_clients = list(self.clients.find({"embedding": {"$exists": True}}))
        best_match = None
        highest_similarity = 0

        for client in all_clients:
            client_embedding = np.array(client['embedding'])
            similarity = compute_sim(client_embedding, embedding)
            if similarity is not None and similarity > Config.CHECK_NEW_CLIENT and similarity > highest_similarity:
                highest_similarity = similarity
                best_match = client

        return best_match, highest_similarity

    def save_attendance_to_api(self, person_id, device_id, image_url, timestamp, score):
        """Send attendance data to FastAPI API"""
        endpoint = "/attendance/create"  # Adjust as per actual API endpoint
        data = {
            'employee_id': person_id,
            'device_id': device_id,
            'timestamp': timestamp,
            'score': score
        }
        try:
            image_response = requests.get(image_url)
            image_response.raise_for_status()
            files = {
                'image': (os.path.basename(image_url), image_response.content, 'image/jpeg')
            }
            send_report(endpoint, data, files)
        except Exception as e:
            Config.logger.error(f"Error sending attendance to API: {e}")

    def update_client_via_api(self, client_id, image_url, timestamp, device_id):
        """Send client visit data to FastAPI API"""
        endpoint = f"/client/visit-history/{client_id}"
        data = {
            'datetime': timestamp,
            'device_id': device_id
        }
        try:
            image_response = requests.get(image_url)
            image_response.raise_for_status()
            files = {
                'image': (os.path.basename(image_url), image_response.content, 'image/jpeg')
            }
            send_report(endpoint, data, files)
        except Exception as e:
            Config.logger.error(f"Error updating client visit via API: {e}")

    def create_client_via_api(self, image_url, first_seen, last_seen, gender, age):
        """Create a new client via FastAPI API"""
        endpoint = "/client/create"
        data = {
            'first_seen': first_seen,
            'last_seen': last_seen,
            'gender': gender,
            'age': age
        }
        try:
            image_response = requests.get(image_url)
            image_response.raise_for_status()
            files = {
                'image': (os.path.basename(image_url), image_response.content, 'image/jpeg')
            }
            send_report(endpoint, data, files)
        except Exception as e:
            Config.logger.error(f"Error creating new client via API: {e}")


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

    def process_faces(self, face_data, image_url, camera_id, timestamp):
        embedding = face_data.embedding

        # Check against employees
        best_match_employee, similarity_employee = self.db.find_matching_employee(embedding)

        if best_match_employee:
            # Employee recognized; send attendance to API
            self.db.save_attendance_to_api(
                person_id=best_match_employee['person_id'],
                device_id=camera_id,
                image_url=image_url,
                timestamp=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                score=face_data.det_score
            )
            Config.logger.info(f"Attendance sent for employee {best_match_employee['person_id']} with similarity {similarity_employee}")
            return

        # Check against clients
        matching_client, similarity_client = self.db.find_matching_client(embedding)

        if matching_client:
            # Existing client; send visit update to API
            self.db.update_client_via_api(
                client_id=matching_client['person_id'],
                image_url=image_url,
                timestamp=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                device_id=camera_id
            )
            Config.logger.info(f"Client {matching_client['person_id']} visit updated with similarity {similarity_client}")
        else:
            # New client; create via API
            # For the purpose of this example, assuming gender and age can be extracted from face_data
            self.db.create_client_via_api(
                image_url=image_url,
                first_seen=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                last_seen=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                gender=int(face_data.gender),
                age=int(face_data.age)
            )
            Config.logger.info(f"New client created from image {image_url}")

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

            timestamp = extract_date_from_filename(os.path.basename(file_path))
            if not timestamp:
                Config.logger.error(f"Could not extract date from filename: {file_path}")
                return

            # Construct image URL based on known pattern
            image_url = f"http://10.30.10.136:8000/uploads/{os.path.basename(file_path)}"

            self.process_faces(face_data, image_url, camera_id, timestamp)

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

            # Instead of saving to MongoDB, send to API
            image_url = f"http://10.30.10.136:8000/uploads/{os.path.basename(image_path)}"
            data = {
                'name': 'Employee Name',  # Replace with actual data
                'email': 'employee@example.com',  # Replace with actual data
                'phone': '1234567890',  # Replace with actual data
                'department_id': 1  # Replace with actual data
            }
            files = {
                'image': (os.path.basename(image_path), open(image_path, 'rb'), 'image/jpeg')
            }
            response = requests.post(
                f"{Config.API_BASE_URL}/employee/create",
                data=data,
                files=files,
                headers=Config.HEADERS
            )
            response.raise_for_status()
            Config.logger.info(f"Employee created via API with ID: {person_id}")
            return True
        except Exception as e:
            Config.logger.error(f"Error adding employee: {e}")
            return False


if __name__ == '__main__':
    load_dotenv()
    images_folder = os.getenv('IMAGES_FOLDER', '/path/to/images')
    runner = MainRunner(images_folder)
    runner.run()
