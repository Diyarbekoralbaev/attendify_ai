# celery_worker.py
import os
from celery import Celery
from dotenv import load_dotenv
from tasks import fetch_and_store_data

load_dotenv()

celery_app = Celery(
    'tasks',
    broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
)

celery_app.conf.beat_schedule = {
    'fetch-data-every-10-minutes': {
        'task': 'tasks.fetch_and_store_data',
        'schedule': 10.0, # 10 seconds for testing
    },
}

celery_app.conf.timezone = 'UTC'

if __name__ == '__main__':
    celery_app.start()
