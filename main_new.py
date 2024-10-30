# main_new.py
import os
import threading
import time
import cv2
import numpy as np
from dotenv import load_dotenv
from insightface.app import FaceAnalysis
from pymongo import MongoClient
import requests
from funcs import compute_sim, extract_date_from_filename, get_faces_data, setup_logger
import faiss
import cProfile

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from tasks import process_image_task, fetch_and_store_data, faiss_index_employee, faiss_index_client

load_dotenv()


class Config:
    CHECK_NEW_CLIENT = float(os.getenv('CHECK_NEW_CLIENT', 0.5))  # Similarity threshold for clients
    EMPLOYEE_SIMILARITY_THRESHOLD = float(os.getenv('EMPLOYEE_SIMILARITY_THRESHOLD', 0.5))  # Similarity threshold for employees
    MIN_DETECTION_CONFIDENCE = float(os.getenv('MIN_DETECTION_CONFIDENCE', 0.6))  # Minimum detection confidence for faces
    DET_SCORE_THRESH = float(os.getenv('DET_SCORE_THRESH', 0.65))
    POSE_THRESHOLD = float(os.getenv('POSE_THRESHOLD', 40))
    logger = setup_logger('MainRunner', 'logs/main.log')
    DIMENSIONS = int(os.getenv('DIMENSIONS', 512))
    DET_SIZE = tuple(map(int, os.getenv('DET_SIZE', '640,640').split(',')))
    API_BASE_URL = os.getenv('API_BASE_URL', 'http://10.30.10.136:8000')


def send_report(endpoint, data, files=None):
    url = f"{Config.API_BASE_URL}{endpoint}"
    try:
        response = requests.post(url, data=data, files=files)
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

        # Initialize Faiss index for employees
        self.employee_embeddings = [np.array(emp['embedding']) for emp in self.employees.find({"embedding": {"$exists": True}})]
        if self.employee_embeddings:
            self.employee_index = faiss.IndexFlatL2(Config.DIMENSIONS)
            self.employee_index.add(np.array(self.employee_embeddings).astype('float32'))
        else:
            self.employee_index = None

        # Similarly, initialize Faiss index for clients
        self.client_embeddings = [np.array(cli['embedding']) for cli in self.clients.find({"embedding": {"$exists": True}})]
        if self.client_embeddings:
            self.client_index = faiss.IndexFlatL2(Config.DIMENSIONS)
            self.client_index.add(np.array(self.client_embeddings).astype('float32'))
        else:
            self.client_index = None

    def find_matching_employee(self, embedding):
        if not self.employee_index:
            return None, 0
        D, I = self.employee_index.search(np.array([embedding]).astype('float32'), k=1)
        similarity = 1 - D[0][0] / (2 * Config.DIMENSIONS)  # Example similarity metric
        if similarity > Config.EMPLOYEE_SIMILARITY_THRESHOLD:
            employee = self.employees.find_one({"embedding": self.employee_embeddings[I[0][0]].tolist()})
            return employee, similarity
        return None, 0

    def find_matching_client(self, embedding):
        if not self.client_index:
            return None, 0
        D, I = self.client_index.search(np.array([embedding]).astype('float32'), k=1)
        similarity = 1 - D[0][0] / (2 * Config.DIMENSIONS)  # Example similarity metric
        if similarity > Config.CHECK_NEW_CLIENT:
            client = self.clients.find_one({"embedding": self.client_embeddings[I[0][0]].tolist()})
            return client, similarity
        return None, 0

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
        self.logger = Config.logger

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
            image_resized = cv2.resize(image, Config.DET_SIZE)

            faces = self.face_processor.process_image(image_resized)

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
        self.logger.info(f"Starting directory observer for: {self.images_folder}")
        event_handler = ImageHandler(self.logger)
        observer = Observer()
        test_camera_dir = os.path.join(self.images_folder, 'test_camera')
        os.makedirs(test_camera_dir, exist_ok=True)
        observer.schedule(event_handler, path=test_camera_dir, recursive=False)
        observer.start()
        try:
            while True:
                time.sleep(10)  # Sleep longer since Watchdog handles events
        except KeyboardInterrupt:
            self.logger.info("Stopping directory observer.")
            observer.stop()
        observer.join()

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
            )
            response.raise_for_status()
            Config.logger.info(f"Employee created via API with ID: {person_id}")
            return True
        except Exception as e:
            Config.logger.error(f"Error adding employee: {e}")
            return False


class ImageHandler(FileSystemEventHandler):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('SNAP.jpg'):
            self.logger.info(f"New image detected: {event.src_path}")
            # Dispatch a Celery task to process the image
            process_image_task.delay(event.src_path, camera_id=1)

if __name__ == '__main__':
    load_dotenv()
    images_folder = os.getenv('IMAGES_FOLDER', '/path/to/images')
    fetch_and_store_data()
    print("Starting main runner...")
    print(f"Total employees: {faiss_index_employee.ntotal}")
    print(f"Total clients: {faiss_index_client.ntotal}")
    runner = MainRunner(images_folder)
    profiler = cProfile.Profile()
    profiler.enable()
    runner.run()
    profiler.disable()
    profiler.dump_stats("main_profiler.stats")
