# tasks.py
import logging
import os
from datetime import datetime
import requests
from celery_worker import celery_app  # Import the central Celery app
from pymongo import MongoClient
from dotenv import load_dotenv
from insightface.app import FaceAnalysis
import cv2
import numpy as np
import faiss
from funcs import compute_sim, setup_logger, extract_date_from_filename, get_faces_data
from bson.binary import Binary
from celery.signals import worker_init

load_dotenv()

# Initialize Logger
logger = setup_logger('CeleryWorker', 'logs/celery_worker.log', level=logging.INFO)

# Initialize MongoDB
mongo_client = MongoClient(os.getenv('MONGODB_LOCAL'))
mongo_db = mongo_client.empl_time_fastapi
employees_collection = mongo_db.employees
clients_collection = mongo_db.clients

# Initialize FaceAnalysis
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Use CPU only
face_app.prepare(ctx_id=-1, det_size=(640, 640))

API_BASE_URL = os.getenv('API_BASE_URL', 'http://10.30.10.136:8000')

# Initialize Faiss indexes with Inner Product for cosine similarity
DIMENSIONS = int(os.getenv('DIMENSIONS', 512))
faiss_index_employee = faiss.IndexFlatIP(DIMENSIONS)
faiss_index_client = faiss.IndexFlatIP(DIMENSIONS)

employee_ids = []
client_ids = []

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


def load_faiss_indexes():
    global faiss_index_employee, faiss_index_client, employee_ids, client_ids
    logger.info("Loading Faiss indexes for employees and clients.")

    # Reset the indexes
    faiss_index_employee = faiss.IndexFlatIP(DIMENSIONS)
    faiss_index_client = faiss.IndexFlatIP(DIMENSIONS)
    employee_ids = []
    client_ids = []

    # Load employee embeddings
    employee_embeddings = []
    for emp in employees_collection.find({"embedding": {"$exists": True}}):
        embedding = np.array(emp['embedding']).astype('float32')
        if embedding.shape[0] != DIMENSIONS:
            logger.warning(f"Employee ID {emp['person_id']} has invalid embedding shape.")
            continue
        embedding = embedding / np.linalg.norm(embedding)  # Normalize for cosine similarity
        employee_embeddings.append(embedding)
        employee_ids.append(emp['person_id'])

    if employee_embeddings:
        employee_embeddings = np.array(employee_embeddings)
        faiss.normalize_L2(employee_embeddings)
        faiss_index_employee.add(employee_embeddings)
        logger.info(f"Loaded {len(employee_embeddings)} employee embeddings into Faiss index.")
    else:
        logger.warning("No employee embeddings loaded into Faiss index.")

    # Load client embeddings
    client_embeddings = []
    for cli in clients_collection.find({"embedding": {"$exists": True}}):
        embedding = np.array(cli['embedding']).astype('float32')
        if embedding.shape[0] != DIMENSIONS:
            logger.warning(f"Client ID {cli['person_id']} has invalid embedding shape.")
            continue
        embedding = embedding / np.linalg.norm(embedding)  # Normalize for cosine similarity
        client_embeddings.append(embedding)
        client_ids.append(cli['person_id'])

    if client_embeddings:
        client_embeddings = np.array(client_embeddings)
        faiss.normalize_L2(client_embeddings)
        faiss_index_client.add(client_embeddings)
        logger.info(f"Loaded {len(client_embeddings)} client embeddings into Faiss index.")
    else:
        logger.warning("No client embeddings loaded into Faiss index.")


@worker_init.connect
def initialize_worker(**kwargs):
    load_faiss_indexes()

@celery_app.task
def fetch_and_store_data():
    global faiss_index_employee, faiss_index_client, employee_ids, client_ids
    logger.info("Starting fetch_and_store_data task")

    try:
        # Fetch Employees
        headers = {'Authorization': f'Bearer {os.getenv("API_TOKEN")}'}
        employees_response = requests.get(f"{API_BASE_URL}/employee/employees", headers=headers)
        employees_response.raise_for_status()
        employees = employees_response.json()

        # Process and store employee embeddings
        for employee in employees:
            image_url = f"{API_BASE_URL}/{employee['image']}"
            embedding = get_embedding_from_url(image_url)
            if embedding is not None:
                embedding = embedding / np.linalg.norm(embedding)  # Normalize
                employees_collection.update_one(
                    {"person_id": employee['id']},
                    {"$set": {
                        "embedding": embedding.tolist(),
                        "name": employee.get('name'),
                        "email": employee.get('email'),
                        "phone": employee.get('phone'),
                        "department_id": employee.get('department_id'),
                        "updated_at": datetime.utcnow()
                    }},
                    upsert=True
                )
                faiss_index_employee.add(np.array([embedding]).astype('float32'))
                employee_ids.append(employee['id'])
                logger.info(f"Stored/Updated embedding for Employee ID: {employee['id']}")
            else:
                logger.error(f"Failed to get embedding for Employee ID: {employee['id']}")

        # # Remove deleted employees from MongoDB
        # for emp in employees_collection.find({"person_id": {"$nin": [emp['id'] for emp in employees]}}):
        #     employees_collection.delete_one({"_id": emp["_id"]})
        #     faiss_index_employee.remove_ids(np.array([emp["person_id"]]))
        #     employee_ids.remove(emp["person_id"])
        #     logger.info(f"Removed deleted employee ID: {emp['person_id']}")

        # Fetch Clients
        clients_response = requests.get(f"{API_BASE_URL}/client/clients", headers=headers)
        clients_response.raise_for_status()
        clients = clients_response.json()

        # Process and store client embeddings
        for client in clients:
            image_url = f"{API_BASE_URL}/{client['image']}"
            embedding = get_embedding_from_url(image_url)
            if embedding is not None:
                embedding = embedding / np.linalg.norm(embedding)  # Normalize
                client_data = clients_collection.find_one({"person_id": client['id']})
                if client_data:
                    # Update existing client
                    clients_collection.update_one(
                        {"person_id": client['id']},
                        {"$set": {
                            "embedding": embedding.tolist(),
                            "first_seen": client.get('first_seen'),
                            "last_seen": client.get('last_seen'),
                            "visit_count": client.get('visit_count', client_data.get('visit_count', 1)),
                            "gender": client.get('gender'),
                            "age": client.get('age'),
                            "updated_at": datetime.utcnow()
                        }},
                        upsert=True
                    )
                else:
                    # Create new client
                    clients_collection.update_one(
                        {"person_id": client['id']},
                        {"$set": {
                            "embedding": embedding.tolist(),
                            "first_seen": client.get('first_seen'),
                            "last_seen": client.get('last_seen'),
                            "visit_count": client.get('visit_count', 1),
                            "age": client.get('age'),
                            "updated_at": datetime.utcnow()
                        }},
                        upsert=True
                    )
                    faiss_index_client.add(np.array([embedding]).astype('float32'))
                    client_ids.append(client['id'])
                    logger.info(f"Stored/Updated embedding for Client ID: {client['id']}")
            else:
                logger.error(f"Failed to get embedding for Client ID: {client['id']}")

        # # Remove deleted clients from MongoDB
        # for cli in clients_collection.find({"person_id": {"$nin": [cli['id'] for cli in clients]}}):
        #     clients_collection.delete_one({"_id": cli["_id"]})
        #     faiss_index_client.remove_ids(np.array([cli["person_id"]]))
        #     client_ids.remove(cli["person_id"])
        #     logger.info(f"Removed deleted client ID: {cli['person_id']}")
        load_faiss_indexes()
    except Exception as e:
        logger.error(f"Error in fetch_and_store_data: {e}")

def get_embedding_from_url(image_url):
    try:
        headers = {'Authorization': f'Bearer {os.getenv("API_TOKEN")}'}
        response = requests.get(image_url, headers=headers)
        response.raise_for_status()
        image_array = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            logger.error(f"Failed to decode image from URL: {image_url}")
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = face_app.get(image_rgb)
        if not faces:
            logger.warning(f"No faces detected in image from URL: {image_url}")
            return None
        # Assuming one face per image for employees/clients
        face = faces[0]
        embedding = face.embedding
        return embedding
    except Exception as e:
        logger.error(f"Error fetching or processing image from URL {image_url}: {e}")
        return None


@celery_app.task
def process_image_task(file_path, camera_id):
    global faiss_index_employee, faiss_index_client, employee_ids, client_ids
    logger.info(f"Processing image: {file_path} from camera_id: {camera_id}")
    try:
        image = cv2.imread(file_path)
        if image is None:
            logger.error(f"Failed to read image from {file_path}")
            return

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (640, 480))

        faces = face_app.get(image_resized)
        logger.info(f"Faces detected: {len(faces)} in image: {file_path}")

        if not faces:
            logger.error(f"No faces found in the image: {file_path}")
            return

        face_data = get_faces_data(faces, min_confidence=Config.MIN_DETECTION_CONFIDENCE)
        if not face_data:
            logger.error(f"Could not extract face data from image: {file_path}")
            return

        timestamp = extract_date_from_filename(os.path.basename(file_path))
        if not timestamp:
            logger.error(f"Could not extract date from filename: {file_path}")
            return

        # Get embedding and normalize
        embedding = face_data.embedding.astype('float32')
        norm = np.linalg.norm(embedding)
        if norm == 0:
            logger.error(f"Embedding norm is zero for image: {file_path}")
            return
        embedding = embedding / norm

        # Search for matching employee first
        print(f"Total employees: {faiss_index_employee.ntotal}")
        if faiss_index_employee.ntotal > 0:
            D_emp, I_emp = faiss_index_employee.search(np.array([embedding]), k=1)
            similarity_emp = float(D_emp[0][0])  # Get cosine similarity directly
            if similarity_emp > Config.EMPLOYEE_SIMILARITY_THRESHOLD:
                employee_id = employee_ids[I_emp[0][0]]
                employee = employees_collection.find_one({"person_id": employee_id})
                if employee:
                    save_attendance_to_api(
                        person_id=employee['person_id'],
                        device_id=camera_id,
                        image_path=file_path,
                        timestamp=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        score=similarity_emp
                    )
                    logger.info(
                        f"Attendance sent for employee {employee['person_id']} with similarity {similarity_emp}")
                    return

        # Then search for matching client
        print(f"Total clients: {faiss_index_client.ntotal}")
        if faiss_index_client.ntotal > 0:
            D_cli, I_cli = faiss_index_client.search(np.array([embedding]), k=1)
            similarity_cli = float(D_cli[0][0])  # Get cosine similarity directly

            logger.info(f"Best client match similarity: {similarity_cli}")
            if similarity_cli > Config.CHECK_NEW_CLIENT:
                client_id = client_ids[I_cli[0][0]]
                client = clients_collection.find_one({"person_id": client_id})
                if client:
                    update_client_via_api(
                        client_id=client['person_id'],
                        datetime=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        device_id=camera_id
                    )
                    logger.info(f"Client {client['person_id']} visit updated with similarity {similarity_cli}")
                    return

        # If no match found, create new client
        new_client_id = create_client_via_api(
            image_path=file_path,
            first_seen=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            last_seen=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            gender=int(face_data.gender),
            age=int(face_data.age)
        )

        if new_client_id:
            # Store the embedding in MongoDB
            clients_collection.update_one(
                {"person_id": new_client_id},
                {"$set": {"embedding": embedding.tolist()}},
                upsert=True
            )

            # Add to Faiss index
            faiss_index_client.add(np.array([embedding]))
            client_ids.append(new_client_id)
            logger.info(f"New client created and indexed with ID: {new_client_id}")
        else:
            logger.error("Failed to create new client")

    except Exception as e:
        logger.error(f"Error processing image {file_path}: {e}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
        bg_file = file_path.replace('SNAP', 'BACKGROUND')
        if os.path.exists(bg_file):
            os.remove(bg_file)

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
        headers = {'Authorization': f'Bearer {os.getenv("API_TOKEN")}'}
        with open(image_path, 'rb') as img_file:
            files = {
                'image': (os.path.basename(image_path), img_file, 'image/jpeg')
            }
            send_report(endpoint, data=data, files=files, headers=headers)
    except Exception as e:
        logger.error(f"Error sending attendance to API: {e}")

def update_client_via_api(client_id, datetime, device_id):
    """Send client visit data to FastAPI API"""
    endpoint = f"/client/visit-history/{client_id}"
    data = {
        'datetime': datetime,
        'device_id': device_id
    }
    try:
        headers = {'Authorization': f'Bearer {os.getenv("API_TOKEN")}'}
        # The /client/visit-history/{client_id} expects JSON data, not multipart/form-data
        send_report_json(endpoint, data=data, headers=headers)
    except Exception as e:
        logger.error(f"Error updating client visit via API: {e}")

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
        headers = {'Authorization': f'Bearer {os.getenv("API_TOKEN")}'}
        with open(image_path, 'rb') as img_file:
            files = {
                'image': (os.path.basename(image_path), img_file, 'image/jpeg')
            }
            response = send_report_with_response(endpoint, data=data, files=files, params=params, headers=headers)
            if response and response.status_code == 200:
                client_data = response.json()
                new_client_id = client_data.get('data').get('id')
                if new_client_id:
                    logger.info(f"New client created with ID: {new_client_id}")
                    return new_client_id
                else:
                    logger.error("New client ID not found in the API response.")
                    return None
            else:
                logger.error(f"Failed to create new client. Status Code: {response.status_code if response else 'No Response'}")
                return None
    except Exception as e:
        logger.error(f"Error creating new client via API: {e}")
        return None

def send_report(endpoint, data=None, files=None, headers=None):
    url = f"{API_BASE_URL}{endpoint}"
    try:
        response = requests.post(url, data=data, files=files, headers=headers)
        response.raise_for_status()
        logger.info(f"Successfully sent report to {endpoint}")
    except requests.RequestException as e:
        logger.error(f"Failed to send report to {endpoint}: {e}")

def send_report_json(endpoint, data=None, headers=None):
    """Send JSON report to FastAPI API"""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        logger.info(f"Successfully sent JSON report to {endpoint}")
    except requests.RequestException as e:
        # Attempt to log the response content for detailed error information
        try:
            error_content = response.json()
            logger.error(f"Failed to send JSON report to {endpoint}: {e}, Response: {error_content}")
        except:
            logger.error(f"Failed to send JSON report to {endpoint}: {e}")

def send_report_with_response(endpoint, data=None, files=None, params=None, headers=None):
    """Send report and return the response object"""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        response = requests.post(url, data=data, files=files, params=params, headers=headers)
        response.raise_for_status()
        logger.info(f"Successfully sent report to {endpoint}")
        return response
    except requests.RequestException as e:
        logger.error(f"Failed to send report to {endpoint}: {e}")
        return None
