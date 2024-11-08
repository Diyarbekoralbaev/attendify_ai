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
import faiss
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime
from funcs import compute_sim, extract_date_from_filename, get_faces_data, setup_logger

load_dotenv()

import asyncio
import websockets
import json

async def websocket_listener(db_manager, face_processor):
    uri = f"{Config.API_BASE_URL.replace('http', 'ws')}/ws"

    async with websockets.connect(uri) as websocket:
        Config.logger.info("Connected to WebSocket server.")
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                Config.logger.info(f"Received data via WebSocket: {data}")

                # Handle the data (e.g., 'employee_update' or 'client_update')
                if data['event'] == 'employee_update':
                    await handle_employee_update(data['data'], db_manager, face_processor)
                elif data['event'] == 'employee_delete':
                    await handle_employee_removed(data['data']['id'], db_manager)
                elif data['event'] == 'client_delete':
                    await handle_client_removed(data['data']['id'], db_manager)
                else:
                    Config.logger.warning(f"Unknown data type received: {data['event']}")

            except websockets.ConnectionClosed:
                Config.logger.error("WebSocket connection closed. Reconnecting...")
                await asyncio.sleep(5)  # Wait before reconnecting
                return await websocket_listener(db_manager, face_processor)
            except Exception as e:
                Config.logger.error(f"Error in WebSocket listener: {e}")
                await asyncio.sleep(1)


async def handle_employee_update(employee_data, db_manager, face_processor):
    person_id = employee_data['id']
    image_url = f"{Config.API_BASE_URL}/{employee_data['image']}"
    embedding = get_embedding_from_url(image_url, face_processor)
    if embedding is not None:
        db_manager.add_employee_embedding(person_id, embedding)
        Config.logger.info(f"Updated embedding for Employee ID: {person_id}")
    else:
        Config.logger.error(f"Failed to get embedding for Employee ID: {person_id}")

async def handle_client_update(client_data, db_manager, face_processor):
    person_id = client_data['id']
    image_url = f"{Config.API_BASE_URL}/{client_data['image']}"
    embedding = get_embedding_from_url(image_url, face_processor)
    if embedding is not None:
        db_manager.add_client_embedding(person_id, embedding)
        Config.logger.info(f"Updated embedding for Client ID: {person_id}")
    else:
        Config.logger.error(f"Failed to get embedding for Client ID: {person_id}")

async def handle_employee_removed(employee_id, db_manager):
    person_id = employee_id
    db_manager.remove_employee_embedding(person_id)
    Config.logger.info(f"Removed embedding for Employee ID: {person_id}")


async def handle_client_removed(client_id, db_manager):
    person_id = client_id
    db_manager.remove_client_embedding(person_id)
    Config.logger.info(f"Removed embedding for Client ID: {person_id}")

# Configuration Class
class Config:
    CHECK_NEW_CLIENT = float(os.getenv('CHECK_NEW_CLIENT', 0.7))  # Adjusted similarity threshold for clients
    EMPLOYEE_SIMILARITY_THRESHOLD = float(os.getenv('EMPLOYEE_SIMILARITY_THRESHOLD', 0.7))  # Adjusted similarity threshold for employees
    MIN_DETECTION_CONFIDENCE = float(os.getenv('MIN_DETECTION_CONFIDENCE', 0.6))  # Minimum detection confidence for faces
    logger = setup_logger('MainRunner', 'logs/main.log')
    DIMENSIONS = int(os.getenv('DIMENSIONS', 512))
    DET_SIZE = tuple(map(int, os.getenv('DET_SIZE', '640,640').split(',')))
    API_BASE_URL = os.getenv('API_BASE_URL', 'http://10.30.10.136:8000')
    API_TOKEN = os.getenv('API_TOKEN', 'your_api_token_here')  # Ensure this is set in your .env
    IMAGES_FOLDER = os.getenv('IMAGES_FOLDER', '/path/to/images')  # Update with your images folder path

    DEFAULT_AGE = int(os.getenv('DEFAULT_AGE', 30))
    DEFAULT_GENDER = int(os.getenv('DEFAULT_GENDER', 0))  # 0 for female, 1 for male

    REPORT_COOLDOWN_SECONDS = int(os.getenv('REPORT_COOLDOWN_SECONDS', 60))  # Cooldown period for sending reports

    # Added POSE_THRESHOLD
    POSE_THRESHOLD = int(os.getenv('POSE_THRESHOLD', 30))  # Pose angle threshold

# Database and Faiss Index Management
class DatabaseManager:
    def __init__(self):
        self.mongo_client = MongoClient(os.getenv('MONGODB_LOCAL'))
        self.mongo_db = self.mongo_client.empl_time_fastapi
        self.employees_collection = self.mongo_db.employees
        self.clients_collection = self.mongo_db.clients

        # Initialize Faiss indexes with Inner Product for cosine similarity
        self.DIMENSIONS = Config.DIMENSIONS
        # Using IndexIDMap to map custom IDs
        self.faiss_index_employee = faiss.IndexIDMap(faiss.IndexFlatIP(self.DIMENSIONS))
        self.faiss_index_client = faiss.IndexIDMap(faiss.IndexFlatIP(self.DIMENSIONS))
        self.lock = threading.Lock()

        # Maintain a mapping from person_id to embedding for compute_sim
        self.employee_embeddings_map = {}
        self.client_embeddings_map = {}

        self.load_faiss_indexes()

    def load_faiss_indexes(self):
        with self.lock:
            Config.logger.info("Loading Faiss indexes for employees and clients.")

            # Reset the indexes
            self.faiss_index_employee = faiss.IndexIDMap(faiss.IndexFlatIP(self.DIMENSIONS))
            self.faiss_index_client = faiss.IndexIDMap(faiss.IndexFlatIP(self.DIMENSIONS))
            self.employee_embeddings_map = {}
            self.client_embeddings_map = {}

            # Load employee embeddings
            employee_embeddings = []
            employee_ids = []
            for emp in self.employees_collection.find({"embedding": {"$exists": True}}):
                embedding = np.array(emp['embedding']).astype('float32')
                if embedding.shape[0] != self.DIMENSIONS:
                    Config.logger.warning(f"Employee ID {emp['person_id']} has invalid embedding shape.")
                    continue
                norm = np.linalg.norm(embedding)
                if norm == 0:
                    Config.logger.warning(f"Employee ID {emp['person_id']} has zero norm embedding.")
                    continue
                embedding = embedding / norm  # Normalize for cosine similarity
                employee_embeddings.append(embedding)
                employee_ids.append(emp['person_id'])
                self.employee_embeddings_map[emp['person_id']] = embedding

            if employee_embeddings:
                employee_embeddings = np.array(employee_embeddings)
                faiss.normalize_L2(employee_embeddings)  # Ensure normalization
                self.faiss_index_employee.add_with_ids(employee_embeddings, np.array(employee_ids))
                Config.logger.info(f"Loaded {len(employee_embeddings)} employee embeddings into Faiss index.")
            else:
                Config.logger.warning("No employee embeddings loaded into Faiss index.")

            # Load client embeddings
            client_embeddings = []
            client_ids = []
            for cli in self.clients_collection.find({"embedding": {"$exists": True}}):
                embedding = np.array(cli['embedding']).astype('float32')
                if embedding.shape[0] != self.DIMENSIONS:
                    Config.logger.warning(f"Client ID {cli['person_id']} has invalid embedding shape.")
                    continue
                norm = np.linalg.norm(embedding)
                if norm == 0:
                    Config.logger.warning(f"Client ID {cli['person_id']} has zero norm embedding.")
                    continue
                embedding = embedding / norm  # Normalize for cosine similarity
                client_embeddings.append(embedding)
                client_ids.append(cli['person_id'])
                self.client_embeddings_map[cli['person_id']] = embedding

            if client_embeddings:
                client_embeddings = np.array(client_embeddings)
                faiss.normalize_L2(client_embeddings)  # Ensure normalization
                self.faiss_index_client.add_with_ids(client_embeddings, np.array(client_ids))
                Config.logger.info(f"Loaded {len(client_embeddings)} client embeddings into Faiss index.")
            else:
                Config.logger.warning("No client embeddings loaded into Faiss index.")

    def add_employee_embedding(self, person_id, embedding):
        with self.lock:
            norm = np.linalg.norm(embedding)
            if norm == 0:
                Config.logger.error(f"Cannot add employee {person_id} with zero norm embedding.")
                return
            embedding = embedding / norm
            self.employees_collection.update_one(
                {"person_id": person_id},
                {"$set": {
                    "embedding": embedding.tolist(),
                    "updated_at": datetime.now()
                }},
                upsert=True
            )
            self.faiss_index_employee.add_with_ids(
                np.array([embedding]).astype('float32'),
                np.array([person_id], dtype='int64')
            )
            self.employee_embeddings_map[person_id] = embedding
            Config.logger.info(f"Stored/Updated embedding for Employee ID: {person_id}")

    def add_client_embedding(self, person_id, embedding):
        with self.lock:
            norm = np.linalg.norm(embedding)
            if norm == 0:
                Config.logger.error(f"Cannot add client {person_id} with zero norm embedding.")
                return
            embedding = embedding / norm
            self.clients_collection.update_one(
                {"person_id": person_id},
                {"$set": {
                    "embedding": embedding.tolist(),
                    "updated_at": datetime.now()
                }},
                upsert=True
            )
            self.faiss_index_client.add_with_ids(
                np.array([embedding]).astype('float32'),
                np.array([person_id], dtype='int64')
            )
            self.client_embeddings_map[person_id] = embedding
            Config.logger.info(f"Stored/Updated embedding for Client ID: {person_id}")

    def remove_employee_embedding(self, person_id):
        with self.lock:
            self.employees_collection.delete_one({"person_id": person_id})
            self.employee_embeddings_map.pop(person_id, None)
            try:
                self.faiss_index_employee.remove_ids(np.array([person_id], dtype='int64'))
                Config.logger.info(f"Removed embedding for Employee ID: {person_id}")
            except Exception as e:
                Config.logger.error(f"Error removing embedding from Faiss index: {e}")

    def remove_client_embedding(self, person_id):
        with self.lock:
            self.clients_collection.delete_one({"person_id": person_id})
            self.client_embeddings_map.pop(person_id, None)
            try:
                self.faiss_index_client.remove_ids(np.array([person_id], dtype='int64'))
                Config.logger.info(f"Removed embedding for Client ID: {person_id}")
            except Exception as e:
                Config.logger.error(f"Error removing embedding from Faiss index: {e}")

    def remove_deleted_employees(self, fetched_employee_ids):
        with self.lock:
            deleted_employees = self.employees_collection.find({"person_id": {"$nin": fetched_employee_ids}})
            deleted_employee_ids = [emp['person_id'] for emp in deleted_employees]

            if deleted_employee_ids:
                try:
                    self.employees_collection.delete_many({"person_id": {"$in": deleted_employee_ids}})
                    Config.logger.info(f"Removed deleted employees: {deleted_employee_ids}")
                    # Rebuild Faiss index
                    self.load_faiss_indexes()
                except Exception as e:
                    Config.logger.error(f"Error removing deleted employees: {e}")

    def remove_deleted_clients(self, fetched_client_ids):
        with self.lock:
            deleted_clients = self.clients_collection.find({"person_id": {"$nin": fetched_client_ids}})
            deleted_client_ids = [cli['person_id'] for cli in deleted_clients]

            if deleted_client_ids:
                try:
                    self.clients_collection.delete_many({"person_id": {"$in": deleted_client_ids}})
                    Config.logger.info(f"Removed deleted clients: {deleted_client_ids}")
                    # Rebuild Faiss index
                    self.load_faiss_indexes()
                except Exception as e:
                    Config.logger.error(f"Error removing deleted clients: {e}")

    def find_matching_employee(self, embedding):
        with self.lock:
            if self.faiss_index_employee.ntotal == 0:
                return None, 0
            D, I = self.faiss_index_employee.search(np.array([embedding]).astype('float32'), k=1)
            if I[0][0] == -1:
                return None, 0
            similarity = float(D[0][0])  # Cosine similarity
            similarity = min(max(similarity, -1.0), 1.0)  # Cap similarity
            Config.logger.debug(f"Faiss similarity: {similarity}")
            if similarity > Config.EMPLOYEE_SIMILARITY_THRESHOLD:
                employee_id = int(I[0][0])
                employee = self.employees_collection.find_one({"person_id": employee_id})
                return employee, similarity
            return None, 0

    def find_matching_client(self, embedding):
        with self.lock:
            if self.faiss_index_client.ntotal == 0:
                return None, 0
            D, I = self.faiss_index_client.search(np.array([embedding]).astype('float32'), k=1)
            if I[0][0] == -1:
                return None, 0
            similarity = float(D[0][0])  # Cosine similarity
            similarity = min(max(similarity, -1.0), 1.0)  # Cap similarity
            Config.logger.debug(f"Faiss similarity: {similarity}")
            if similarity > Config.CHECK_NEW_CLIENT:
                client_id = int(I[0][0])
                client = self.clients_collection.find_one({"person_id": client_id})
                return client, similarity
            return None, 0

# Face Analysis and Processing
class FaceProcessor:
    def __init__(self):
        # Initialize FaceAnalysis with desired models
        self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])  # Use CUDA if available
        self.app.prepare(ctx_id=0, det_size=Config.DET_SIZE)

    def get_embedding_from_image(self, image):
        faces = self.app.get(image)
        if not faces:
            return None, None, None
        # Get the face with the highest detection score
        face = get_faces_data(faces, min_confidence=Config.MIN_DETECTION_CONFIDENCE)
        if face:
            # Pose check
            if abs(face.pose[1]) > Config.POSE_THRESHOLD or abs(face.pose[0]) > Config.POSE_THRESHOLD:
                Config.logger.warning(f"Face pose exceeds threshold: pose={face.pose}")
                return None, None, None

            embedding = face.embedding
            # Normalize embedding
            norm = np.linalg.norm(embedding)
            Config.logger.debug(f"Embedding norm: {norm}")
            if norm == 0:
                Config.logger.warning("Detected face has zero norm embedding.")
                return None, None, None
            embedding = embedding / norm
            Config.logger.debug(f"Normalized embedding: {embedding}")
            age = getattr(face, 'age', None)
            gender = getattr(face, 'gender', None)
            return embedding, age, gender
        return None, None, None


# API Interaction Functions
def save_attendance_to_api(person_id, device_id, image_path, timestamp, score):
    """Send attendance data to FastAPI API"""
    endpoint = "/attendance/create"  # Adjust as per actual API endpoint
    data = {
        'employee_id': person_id,
        'device_id': device_id,
        'timestamp': timestamp,
        'score': score
    }
    try:
        headers = {'Authorization': f'Bearer {Config.API_TOKEN}'}
        with open(image_path, 'rb') as img_file:
            files = {
                'image': (os.path.basename(image_path), img_file, 'image/jpeg')
            }
            response = send_report(endpoint, data=data, files=files, headers=headers)
            if response:
                Config.logger.info(f"Attendance sent for employee {person_id} with similarity {score}")
    except Exception as e:
        Config.logger.error(f"Error sending attendance to API: {e}")

def update_client_via_api(client_id, datetime_str, device_id):
    """Send client visit data to FastAPI API"""
    endpoint = f"/client/visit-history/{client_id}"
    data = {
        'datetime': datetime_str,
        'device_id': device_id
    }
    try:
        headers = {'Authorization': f'Bearer {Config.API_TOKEN}'}
        response = send_report_json(endpoint, data=data, headers=headers)
        if response:
            Config.logger.info(f"Client {client_id} visit updated.")
    except Exception as e:
        Config.logger.error(f"Error updating client visit via API: {e}")

def create_client_via_api(image_path, first_seen, last_seen, gender, age):
    """Create a new client via FastAPI API and return the new client ID"""
    endpoint = "/client/create"
    data = {
        'first_seen': first_seen,
        'last_seen': last_seen
    }
    # Query Parameters
    params = {
        'visit_count': 1,
        'gender': gender,
        'age': age
    }
    try:
        headers = {'Authorization': f'Bearer {Config.API_TOKEN}'}
        with open(image_path, 'rb') as img_file:
            files = {
                'image': (os.path.basename(image_path), img_file, 'image/jpeg')
            }
            response = send_report_with_response(endpoint, data=data, files=files, params=params, headers=headers)
            if response and response.status_code == 200:
                client_data = response.json()
                new_client_id = client_data.get('data', {}).get('id')
                if new_client_id:
                    Config.logger.info(f"New client created with ID: {new_client_id}")
                    return new_client_id
                else:
                    Config.logger.error("New client ID not found in the API response.")
                    return None
            else:
                Config.logger.error(f"Failed to create new client. Status Code: {response.status_code if response else 'No Response'}")
                return None
    except Exception as e:
        Config.logger.error(f"Error creating new client via API: {e}")
        return None

def send_report(endpoint, data=None, files=None, headers=None):
    url = f"{Config.API_BASE_URL}{endpoint}"
    try:
        response = requests.post(url, data=data, files=files, headers=headers)
        response.raise_for_status()
        Config.logger.info(f"Successfully sent report to {endpoint}")
        return response
    except requests.RequestException as e:
        Config.logger.error(f"Failed to send report to {endpoint}: {e}")
        return None

def send_report_json(endpoint, data=None, headers=None):
    """Send JSON report to FastAPI API"""
    url = f"{Config.API_BASE_URL}{endpoint}"
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        Config.logger.info(f"Successfully sent JSON report to {endpoint}")
        return response
    except requests.RequestException as e:
        # Attempt to log the response content for detailed error information
        try:
            error_content = response.json()
            Config.logger.error(f"Failed to send JSON report to {endpoint}: {e}, Response: {error_content}")
        except Exception:
            Config.logger.error(f"Failed to send JSON report to {endpoint}: {e}")
        return None

def send_report_with_response(endpoint, data=None, files=None, params=None, headers=None):
    """Send report and return the response object"""
    url = f"{Config.API_BASE_URL}{endpoint}"
    try:
        response = requests.post(url, data=data, files=files, params=params, headers=headers)
        response.raise_for_status()
        Config.logger.info(f"Successfully sent report to {endpoint}")
        return response
    except requests.RequestException as e:
        Config.logger.error(f"Failed to send report to {endpoint}: {e}")
        return None

# Image Processing Function
def process_image(file_path, camera_id, db_manager, face_processor, employee_last_report_times, client_last_report_times, lock):
    Config.logger.info(f"Processing image: {file_path} from camera_id: {camera_id}")
    try:
        image = cv2.imread(file_path)
        if image is None:
            Config.logger.error(f"Failed to read image from {file_path}")
            return

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, Config.DET_SIZE)

        embedding, age, gender = face_processor.get_embedding_from_image(image_resized)
        if embedding is None:
            Config.logger.error(f"No face embedding found in image: {file_path}")
            return

        if age is None:
            age = Config.DEFAULT_AGE
        else:
            age = int(round(age))

        if gender is None:
            gender = Config.DEFAULT_GENDER
        else:
            gender = int(round(gender))

        timestamp = extract_date_from_filename(os.path.basename(file_path))
        if not timestamp:
            Config.logger.error(f"Could not extract date from filename: {file_path}")
            return

        # Search for matching employee
        employee, similarity_emp = db_manager.find_matching_employee(embedding)
        if employee:
            person_id = employee['person_id']
            with lock:
                last_report_time = employee_last_report_times.get(person_id)
                current_time = datetime.now()
                if last_report_time and (current_time - last_report_time).total_seconds() < Config.REPORT_COOLDOWN_SECONDS:
                    Config.logger.info(f"Employee {person_id} was seen {current_time - last_report_time} ago. Skipping attendance report.")
                    return
                else:
                    save_attendance_to_api(
                        person_id=employee['person_id'],
                        device_id=camera_id,
                        image_path=file_path,
                        timestamp=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        score=similarity_emp
                    )
                    employee_last_report_times[person_id] = current_time
                    return

        # Search for matching client
        client, similarity_cli = db_manager.find_matching_client(embedding)
        if client:
            person_id = client['person_id']
            with lock:
                last_report_time = client_last_report_times.get(person_id)
                current_time = datetime.now()
                if last_report_time and (current_time - last_report_time).total_seconds() < Config.REPORT_COOLDOWN_SECONDS:
                    Config.logger.info(f"Client {person_id} was seen {current_time - last_report_time} ago. Skipping visit history update.")
                    return
                else:
                    # Send visit history update
                    update_client_via_api(
                        client_id=person_id,
                        datetime_str=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        device_id=camera_id
                    )
                    # Update last report time
                    client_last_report_times[person_id] = current_time
                    Config.logger.info(f"Client {person_id} visited with similarity {similarity_cli}")
            return

        # If no match found, create new client
        new_client_id = create_client_via_api(
            image_path=file_path,
            first_seen=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            last_seen=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            gender=gender,
            age=age
        )

        if new_client_id:
            # Store the embedding in MongoDB
            db_manager.add_client_embedding(new_client_id, embedding)
        else:
            Config.logger.error("Failed to create new client")

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

# Fetch and Store Data Function
def fetch_and_store_data(db_manager, face_processor):
    Config.logger.info("Starting fetch_and_store_data task")

    try:
        headers = {'Authorization': f'Bearer {Config.API_TOKEN}'}
        # Fetch Employees
        employees_response = requests.get(f"{Config.API_BASE_URL}/employee/employees", headers=headers)
        employees_response.raise_for_status()
        employees = employees_response.json()

        fetched_employee_ids = [emp['id'] for emp in employees]

        # Process and store employee embeddings
        for employee in employees:
            image_url = f"{Config.API_BASE_URL}/{employee['image']}"
            embedding = get_embedding_from_url(image_url, face_processor)
            if embedding is not None:
                db_manager.add_employee_embedding(employee['id'], embedding)
            else:
                Config.logger.error(f"Failed to get embedding for Employee ID: {employee['id']}")

        # Identify and remove deleted employees from MongoDB
        db_manager.remove_deleted_employees(fetched_employee_ids)

        # Fetch Clients
        clients_response = requests.get(f"{Config.API_BASE_URL}/client/clients", headers=headers)
        clients_response.raise_for_status()
        clients = clients_response.json()

        fetched_client_ids = [cli['id'] for cli in clients]

        # Process and store client embeddings
        for client in clients:
            image_url = f"{Config.API_BASE_URL}/{client['image']}"
            embedding = get_embedding_from_url(image_url, face_processor)
            if embedding is not None:
                db_manager.add_client_embedding(client['id'], embedding)
                Config.logger.info(f"Stored/Updated embedding for Client ID: {client['id']}")
            else:
                Config.logger.error(f"Failed to get embedding for Client ID: {client['id']}")

        # Identify and remove deleted clients from MongoDB
        db_manager.remove_deleted_clients(fetched_client_ids)

        Config.logger.info("fetch_and_store_data task completed successfully.")
    except Exception as e:
        Config.logger.error(f"Error in fetch_and_store_data: {e}")

def get_embedding_from_url(image_url, face_processor):
    try:
        headers = {'Authorization': f'Bearer {Config.API_TOKEN}'}
        response = requests.get(image_url, headers=headers)
        response.raise_for_status()
        image_array = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            Config.logger.error(f"Failed to decode image from URL: {image_url}")
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        embedding, age, gender = face_processor.get_embedding_from_image(image_rgb)
        if embedding is None:
            Config.logger.warning(f"No faces detected or pose exceeds threshold in image from URL: {image_url}")
            return None
        return embedding
    except Exception as e:
        Config.logger.error(f"Error fetching or processing image from URL {image_url}: {e}")
        return None

# Image Handler for Watchdog
class ImageHandler(FileSystemEventHandler):
    def __init__(self, executor, db_manager, face_processor, logger, employee_last_report_times, client_last_report_times, lock):
        super().__init__()
        self.executor = executor
        self.db_manager = db_manager
        self.face_processor = face_processor
        self.logger = logger
        self.employee_last_report_times = employee_last_report_times
        self.client_last_report_times = client_last_report_times
        self.lock = lock

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('SNAP.jpg'):
            self.logger.info(f"New image detected: {event.src_path}")
            # Dispatch a thread to process the image
            self.executor.submit(
                process_image,
                event.src_path,
                camera_id=1,
                db_manager=self.db_manager,
                face_processor=self.face_processor,
                employee_last_report_times=self.employee_last_report_times,
                client_last_report_times=self.client_last_report_times,
                lock=self.lock
            )

# Main Runner Class
class MainRunner:
    def __init__(self, images_folder):
        self.images_folder = images_folder
        self.db_manager = DatabaseManager()
        self.face_processor = FaceProcessor()
        self.executor = ThreadPoolExecutor(max_workers=10)  # Adjust the number of workers as needed
        self.logger = Config.logger
        self.employee_last_report_times = {}
        self.client_last_report_times = {}
        self.lock = threading.Lock()

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def run(self):
        self.logger.info(f"Starting directory observer for: {self.images_folder}")
        event_handler = ImageHandler(
            self.executor,
            self.db_manager,
            self.face_processor,
            self.logger,
            self.employee_last_report_times,
            self.client_last_report_times,
            self.lock
        )
        observer = Observer()
        test_camera_dir = os.path.join(self.images_folder, 'test_camera')
        os.makedirs(test_camera_dir, exist_ok=True)
        observer.schedule(event_handler, path=test_camera_dir, recursive=False)
        observer.start()

        # Start the periodic fetch_and_store_data in a separate thread
        fetch_thread = threading.Thread(target=fetch_and_store_data, args=(self.db_manager, self.face_processor), daemon=True)
        fetch_thread.start()

        # Start the WebSocket listener in a separate thread
        ws_thread = threading.Thread(target=self.start_websocket_listener, daemon=True)
        logging.info("Starting WebSocket listener.")
        ws_thread.start()

        try:
            while True:
                time.sleep(1)  # Keep the main thread alive
        except KeyboardInterrupt:
            self.logger.info("Stopping directory observer.")
            observer.stop()
        observer.join()
        self.executor.shutdown(wait=True)

    def start_websocket_listener(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(websocket_listener(self.db_manager, self.face_processor))

# Entry Point
if __name__ == '__main__':
    images_folder = Config.IMAGES_FOLDER
    runner = MainRunner(images_folder)
    runner.run()
