# camera_capture_headless.py
import cv2
import os
import numpy as np
import logging
import threading
import queue
import concurrent.futures
from datetime import datetime, timedelta
from insightface.app import FaceAnalysis
from deep_sort_realtime.deepsort_tracker import DeepSort
from dotenv import load_dotenv
import time

load_dotenv()

# Setup logging
logging.basicConfig(filename='camera_capture.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Camera credentials and URLs
camera_username = 'admin'
camera_password = 'Qwerty12'
camera_ip = '109.94.174.13'

video_url = f'rtsp://{camera_username}:{camera_password}@{camera_ip}/Streaming/Channels/1001'
phone_camera_url = "rtsp://admin:ClaySec25@192.168.1.20/Streaming/Channels/101"

class CameraCapture:
    def __init__(self, save_path, min_detection_confidence=0.7, min_quality_score=200.0):
        self.save_path = save_path
        self.min_detection_confidence = min_detection_confidence
        self.min_quality_score = min_quality_score

        # Face analysis setup
        self.face_analyzer = FaceAnalysis(name='buffalo_s', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.face_analyzer.prepare(ctx_id=0)

        # Deep SORT tracker
        self.tracker = DeepSort(max_age=30,
                                n_init=3,
                                nn_budget=100,
                                max_cosine_distance=0.3)

        # Create directories
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, 'test_camera'), exist_ok=True)

        # Initialize camera
        self.camera = self.initialize_camera()

        # Face recognition variables
        self.face_embeddings = {}
        self.last_capture_times = {}
        self.person_id_counter = 0

        # Cooldown settings
        self.cooldown_period = timedelta(minutes=1)
        self.embedding_similarity_threshold = 0.5

        # Frame queue for threading
        self.frame_queue = queue.Queue(maxsize=5)

        # Distance estimation parameters
        self.reference_face_area = 6000
        self.reference_distance = 8.0
        self.focal_length = 800

        # Thread pool for asynchronous saving
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        # Frame processing control
        self.process_every_n_frames = 2
        self.frame_count = 0

    def initialize_camera(self, max_retries=3):
        """Initialize camera with retries."""
        for attempt in range(max_retries):
            try:
                logging.info(f"Attempting to initialize IP camera, attempt {attempt + 1}")
                camera = cv2.VideoCapture(phone_camera_url)

                if camera.isOpened():
                    ret, frame = camera.read()
                    if ret and frame is not None:
                        logging.info("Successfully connected to IP camera.")
                        time.sleep(0.5)
                        return camera

                camera.release()
                time.sleep(1)

            except Exception as e:
                logging.error(f"Error initializing IP camera: {e}")
                continue

        raise ValueError("Failed to connect to the IP camera")

    def generate_filename(self):
        """Generate a unique filename based on timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"camera_1_{timestamp}"

    def save_detection(self, frame, face_frame, person_id):
        """Save detected face and background frames."""
        filename = self.generate_filename()
        save_dir = os.path.join(self.save_path, 'test_camera')

        background_path = os.path.join(save_dir, f"{filename}_BACKGROUND.jpg")
        cv2.imwrite(background_path, frame)

        snap_path = os.path.join(save_dir, f"{filename}_SNAP.jpg")
        cv2.imwrite(snap_path, face_frame)

        logging.info(f"Saved detection of person {person_id} to {snap_path}")

    def save_detection_async(self, frame, face_frame, person_id):
        """Asynchronously save detection to avoid blocking."""
        self.executor.submit(self.save_detection, frame, face_frame, person_id)

    def calculate_embedding_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between embeddings."""
        if embedding1 is None or embedding2 is None:
            return 0
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def calculate_image_sharpness(self, image):
        """Calculate image sharpness using Laplacian variance."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def estimate_distance(self, bbox_width, bbox_height):
        """
        Estimate the distance of the face from the camera using the pinhole camera model.
        distance = (focal_length * real_face_width) / face_width_in_pixels
        """
        real_face_width = 0.16  # meters
        face_width_pixels = (bbox_width + bbox_height) / 2

        if face_width_pixels == 0:
            return float('inf')

        distance = (self.focal_length * real_face_width) / face_width_pixels
        return distance

    def enhance_sharpness(self, image):
        """Enhance image sharpness using a sharpening kernel."""
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened

    def process_frame(self):
        while True:
            try:
                frame = self.frame_queue.get()
                if frame is None:
                    break

                self.frame_count += 1
                if self.frame_count % self.process_every_n_frames != 0:
                    continue  # Skip frame

                original_height, original_width = frame.shape[:2]
                # resize frame for max quality
                resized_frame = cv2.resize(frame, (320, 240))

                faces = self.face_analyzer.get(resized_frame)

                detections = []
                for face in faces:
                    if face.det_score < self.min_detection_confidence:
                        continue

                    yaw, pitch, roll = face.pose

                    # Set thresholds for yaw and pitch
                    yaw_threshold = 45  # degrees
                    pitch_threshold = 45  # degrees
                    roll_threshold = 45  # degrees

                    # Check if the face is frontal
                    if (abs(yaw) > yaw_threshold or
                            abs(pitch) > pitch_threshold or
                            abs(roll) > roll_threshold):
                        # Skip faces that are not frontal
                        # logging.info(f"Skipping face with yaw: {yaw:.2f}, pitch: {pitch:.2f}, roll: {roll:.2f}")
                        continue

                    bbox = face.bbox.astype(int)
                    x, y, x2, y2 = bbox
                    w = x2 - x
                    h = y2 - y

                    # Estimate distance
                    distance = self.estimate_distance(w, h)

                    # Only proceed if within reference distance
                    if distance > self.reference_distance:
                        continue

                    detections.append(([x, y, w, h], face.det_score, face.embedding))

                # Update tracker with current detections
                tracks = self.tracker.update_tracks(detections, frame=resized_frame)

                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    track_id = track.track_id
                    bbox = track.to_ltrb()
                    x1, y1, x2, y2 = map(int, bbox)

                    # Map bbox back to original frame size
                    scale_x = original_width / 320
                    scale_y = original_height / 240
                    x1_orig = int(x1 * scale_x)
                    y1_orig = int(y1 * scale_y)
                    x2_orig = int(x2 * scale_x)
                    y2_orig = int(y2 * scale_y)

                    # Extract the face region with margin
                    margin = 100
                    y1_m, y2_m = max(0, y1_orig - margin), min(original_height, y2_orig + margin)
                    x1_m, x2_m = max(0, x1_orig - margin), min(original_width, x2_orig + margin)
                    face_frame = frame[y1_m:y2_m, x1_m:x2_m]  # Using original frame

                    # Calculate sharpness
                    sharpness = self.calculate_image_sharpness(face_frame)

                    if sharpness < self.min_quality_score:
                        continue

                    # Check cooldown
                    current_time = datetime.now()
                    last_capture = self.last_capture_times.get(track_id, datetime.min)
                    if (current_time - last_capture) >= self.cooldown_period:
                        self.save_detection_async(frame, face_frame, track_id)
                        self.last_capture_times[track_id] = current_time
                        logging.info(f"Captured and saved detection for person {track_id}")
                    else:
                        logging.info(f"Person {track_id} is within cooldown period. Skipping capture.")

                # Since we're running headless, we don't need to display the frame

            except Exception as e:
                logging.error(f"Error processing frame: {e}")
                time.sleep(0.1)

    def frame_grabber(self):
        while True:
            ret, frame = self.camera.read()
            if not ret:
                logging.warning("Frame grab failed. Reinitializing camera...")
                self.camera = self.initialize_camera()
                continue

            if not self.frame_queue.full():
                self.frame_queue.put(frame)

        self.frame_queue.put(None)

    def run(self):
        grabber_thread = threading.Thread(target=self.frame_grabber, daemon=True)
        grabber_thread.start()
        self.process_frame()
        grabber_thread.join()
        self.executor.shutdown(wait=True)
        # No need to call cv2.destroyAllWindows() since we are not creating any windows

if __name__ == "__main__":
    try:
        capture = CameraCapture(save_path="./images", min_quality_score=250.0)  # Adjusted threshold
        capture.run()
    except Exception as e:
        logging.fatal(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
