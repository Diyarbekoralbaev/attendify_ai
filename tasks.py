# tasks.py
import os
from datetime import datetime
import requests
from celery import Celery
from pymongo import MongoClient
from dotenv import load_dotenv
from insightface.app import FaceAnalysis
import cv2
import numpy as np
from funcs import compute_sim, setup_logger

load_dotenv()

celery_app = Celery(
    'tasks',
    broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
)

# Initialize MongoDB
mongo_client = MongoClient(os.getenv('MONGODB_LOCAL'))
mongo_db = mongo_client.empl_time_fastapi
employees_collection = mongo_db.employees
clients_collection = mongo_db.clients

# Initialize Logger
logger = setup_logger('CeleryWorker', 'logs/celery_worker.log')

# Initialize FaceAnalysis
face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

API_BASE_URL = os.getenv('API_BASE_URL', 'http://10.30.10.136:8000')

@celery_app.task
def fetch_and_store_data():
    logger.info("Starting fetch_and_store_data task")

    try:
        # Fetch Employees
        employees_response = requests.get(f"{API_BASE_URL}/employee/employees")
        employees_response.raise_for_status()
        employees = employees_response.json()

        # Process and store employee embeddings
        for employee in employees:
            image_url = f"{API_BASE_URL}/{employee['image']}"
            embedding = get_embedding_from_url(image_url)
            if embedding is not None:
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
                logger.info(f"Stored/Updated embedding for Employee ID: {employee['id']}")
            else:
                logger.error(f"Failed to get embedding for Employee ID: {employee['id']}")

        # Fetch Clients
        clients_response = requests.get(f"{API_BASE_URL}/client/clients")
        clients_response.raise_for_status()
        clients = clients_response.json()

        # Process and store client embeddings
        for client in clients:
            image_url = f"{API_BASE_URL}/{client['image']}"
            embedding = get_embedding_from_url(image_url)
            if embedding is not None:
                clients_collection.update_one(
                    {"person_id": client['id']},
                    {"$set": {
                        "embedding": embedding.tolist(),
                        "first_seen": client.get('first_seen'),
                        "last_seen": client.get('last_seen'),
                        "visit_count": client.get('visit_count', 1),
                        "gender": client.get('gender'),
                        "age": client.get('age'),
                        "updated_at": datetime.utcnow()
                    }},
                    upsert=True
                )
                logger.info(f"Stored/Updated embedding for Client ID: {client['id']}")
            else:
                logger.error(f"Failed to get embedding for Client ID: {client['id']}")

    except Exception as e:
        logger.error(f"Error in fetch_and_store_data: {e}")

def get_embedding_from_url(image_url):
    try:
        response = requests.get(image_url)
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
