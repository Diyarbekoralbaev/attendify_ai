#camera_capture.py
import cv2
import os
from datetime import datetime, timedelta
import numpy as np
from insightface.app import FaceAnalysis
import time
from dotenv import load_dotenv
import threading
import queue
import logging
from deep_sort_realtime.deepsort_tracker import DeepSort
import concurrent.futures

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
        self.face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
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
                resized_frame = cv2.resize(frame, (320, 240))  # Reduced resolution

                # Optionally enhance sharpness
                # resized_frame = self.enhance_sharpness(resized_frame)

                faces = self.face_analyzer.get(resized_frame)

                detections = []
                for face in faces:
                    if face.det_score < self.min_detection_confidence:
                        continue

                    bbox = face.bbox.astype(int)
                    x, y, x2, y2 = bbox
                    w = x2 - x
                    h = y2 - y

                    # Estimate distance
                    distance = self.estimate_distance(w, h)
                    # logging.info(f"Estimated distance for face: {distance:.2f} meters")

                    # Only proceed if within 3 meters
                    if distance > self.reference_distance:
                        # logging.info(f"Face at {distance:.2f} meters is outside the 3-meter range. Skipping.")
                        continue

                    detections.append(([x, y, w, h], face.det_score, face.embedding))

                # Update tracker with current detections
                tracks = self.tracker.update_tracks(detections, frame=resized_frame)

                display_frame = frame.copy()
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
                    margin = 100  # Reduced margin for efficiency
                    y1_m, y2_m = max(0, y1_orig - margin), min(original_height, y2_orig + margin)
                    x1_m, x2_m = max(0, x1_orig - margin), min(original_width, x2_orig + margin)
                    face_frame = display_frame[y1_m:y2_m, x1_m:x2_m]

                    # Calculate sharpness
                    sharpness = self.calculate_image_sharpness(face_frame)
                    # logging.info(f"Calculated sharpness for track {track_id}: {sharpness:.2f}")

                    if sharpness < self.min_quality_score:
                        # logging.info(f"Face too blurry (sharpness: {sharpness:.2f}). Skipping.")
                        continue

                    # Check cooldown
                    current_time = datetime.now()
                    last_capture = self.last_capture_times.get(track_id, datetime.min)
                    if (current_time - last_capture) >= self.cooldown_period:
                        self.save_detection_async(display_frame, face_frame, track_id)
                        self.last_capture_times[track_id] = current_time
                        color = (0, 255, 0)  # Green for new capture
                    else:
                        color = (0, 0, 255)  # Red if within cooldown

                    # Draw bounding box and label
                    cv2.rectangle(display_frame, (x1_orig, y1_orig), (x2_orig, y2_orig), color, 2)
                    cv2.putText(display_frame, f"Person {track_id}", (x1_orig, y1_orig - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Display the number of faces detected within 3 meters
                cv2.putText(display_frame, f"Faces within 3m: {len(tracks)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Camera Feed', display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("Exit signal received. Shutting down.")
                    break

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
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        capture = CameraCapture(save_path="./images", min_quality_score=150.0)  # Adjusted threshold
        capture.run()
    except Exception as e:
        logging.fatal(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
        cv2.destroyAllWindows()
