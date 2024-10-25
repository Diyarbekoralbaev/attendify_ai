import cv2
import os
from datetime import datetime, timedelta
import numpy as np
from insightface.app import FaceAnalysis
import time
from dotenv import load_dotenv
import threading
import queue

load_dotenv()

camera_username = os.getenv('CAMERA_USERNAME')
camera_password = os.getenv('CAMERA_PASSWORD')

camera_ip = os.getenv('CAMERA_IP')

video_url = f'rtsp://{camera_username}:{camera_password}@{camera_ip}/Streaming/Channels/101'


class CameraCapture:
    def __init__(self, save_path, min_detection_confidence=0.65, min_quality_score=100.0):
        self.save_path = save_path
        self.min_detection_confidence = min_detection_confidence
        self.min_quality_score = min_quality_score  # New parameter for quality threshold
        self.face_analyzer = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.face_analyzer.prepare(ctx_id=0)  # Adjusted NMS threshold for better detection

        # Create directories
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, 'test_camera'), exist_ok=True)

        # Initialize camera with retries
        self.camera = self.initialize_camera()

        # Face recognition variables
        self.face_embeddings = {}  # Mapping from person_id to embedding
        self.last_capture_times = {}  # Mapping from person_id to last capture time
        self.person_id_counter = 0  # Counter to assign unique person IDs

        # Cooldown settings
        self.cooldown_period = timedelta(minutes=5)  # Adjust cooldown time as needed
        self.embedding_similarity_threshold = 0.6  # Threshold for face similarity

        # Frame queue for threading
        self.frame_queue = queue.Queue(maxsize=5)

    def initialize_camera(self, max_retries=3):
        """Initialize camera using IP camera stream"""
        for attempt in range(max_retries):
            try:
                print(f"Attempting to initialize IP camera, attempt {attempt + 1}")
                camera = cv2.VideoCapture(0)

                if camera.isOpened():
                    ret, frame = camera.read()
                    if ret and frame is not None:
                        print(f"Successfully connected to IP camera.")
                        time.sleep(0.5)
                        return camera

                camera.release()
                time.sleep(1)

            except Exception as e:
                print(f"Error initializing IP camera: {e}")
                continue

        raise ValueError("Failed to connect to the IP camera")

    def generate_filename(self):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"camera_1_{timestamp}"

    def save_detection(self, frame, face_frame, face_embedding, person_id):
        filename = self.generate_filename()
        save_dir = os.path.join(self.save_path, 'test_camera')

        background_path = os.path.join(save_dir, f"{filename}_BACKGROUND.jpg")
        cv2.imwrite(background_path, frame)

        snap_path = os.path.join(save_dir, f"{filename}_SNAP.jpg")
        cv2.imwrite(snap_path, face_frame)

        print(f"Saved detection of person {person_id} to {snap_path}")

    def calculate_embedding_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between face embeddings"""
        if embedding1 is None or embedding2 is None:
            return 0
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def calculate_image_sharpness(self, image):
        """Calculate the sharpness of an image using the variance of the Laplacian"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var

    def process_frame(self):
        while True:
            try:
                frame = self.frame_queue.get()
                if frame is None:
                    break  # Exit if None is received

                display_frame = frame.copy()
                frame = cv2.resize(frame, (640, 480))

                # Detect faces
                faces = self.face_analyzer.get(frame)

                # Process each face
                for face in faces:
                    if face.det_score < self.min_detection_confidence:
                        continue

                    embedding = face.embedding
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox

                    # Extract face region with margin
                    margin = 20
                    y1_m = max(0, y1 - margin)
                    y2_m = min(frame.shape[0], y2 + margin)
                    x1_m = max(0, x1 - margin)
                    x2_m = min(frame.shape[1], x2 + margin)
                    face_frame = frame[y1_m:y2_m, x1_m:x2_m]

                    # Check image quality
                    sharpness = self.calculate_image_sharpness(face_frame)
                    if sharpness < self.min_quality_score:
                        print(f"Face is too blurry (sharpness: {sharpness:.2f}). Skipping.")
                        continue  # Skip processing this face

                    # Compare embedding to existing embeddings
                    person_id = None
                    max_similarity = 0
                    for pid, emb in self.face_embeddings.items():
                        similarity = self.calculate_embedding_similarity(embedding, emb)
                        if similarity > max_similarity:
                            max_similarity = similarity
                            person_id = pid

                    if max_similarity < self.embedding_similarity_threshold:
                        # Assign new person ID
                        person_id = self.person_id_counter
                        self.person_id_counter += 1
                        # Store the embedding
                        self.face_embeddings[person_id] = embedding
                        # Initialize last capture time
                        self.last_capture_times[person_id] = datetime.min
                        captured = False
                    else:
                        # Update the embedding
                        self.face_embeddings[person_id] = embedding
                        captured = (datetime.now() - self.last_capture_times[person_id]) < self.cooldown_period

                    # Check cooldown
                    if not captured:
                        self.save_detection(frame, face_frame, embedding, person_id)
                        # Update last capture time
                        self.last_capture_times[person_id] = datetime.now()
                        color = (0, 255, 0)  # Green for new capture
                    else:
                        color = (0, 0, 255)  # Red for cooldown

                    # Draw rectangle around face
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame,
                                f"Person {person_id}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2)

                # Display number of detected faces
                cv2.putText(display_frame,
                            f"Faces detected: {len(faces)}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2)

                # Show frame
                cv2.imshow('Camera Feed', display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"Error in processing thread: {e}")
                time.sleep(0.1)

    def frame_grabber(self):
        frame_failure_count = 0
        max_frame_failures = 10

        try:
            while True:
                ret, frame = self.camera.read()

                if not ret or frame is None:
                    frame_failure_count += 1
                    print(f"Failed to grab frame. Attempt {frame_failure_count}/{max_frame_failures}")

                    if frame_failure_count >= max_frame_failures:
                        print("Too many frame grab failures. Reinitializing camera...")
                        self.camera.release()
                        self.camera = self.initialize_camera()
                        frame_failure_count = 0

                    time.sleep(0.1)
                    continue

                frame_failure_count = 0

                # Put frame into the queue
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                else:
                    # Drop the frame if the queue is full
                    pass

        finally:
            self.camera.release()
            self.frame_queue.put(None)  # Signal the processing thread to exit

    def run(self):
        # Start the frame grabber thread
        grabber_thread = threading.Thread(target=self.frame_grabber)
        grabber_thread.start()

        # Start the frame processing in the main thread
        self.process_frame()

        # Wait for the grabber thread to finish
        grabber_thread.join()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        capture = CameraCapture(save_path="./images", min_quality_score=150.0)  # Adjust min_quality_score as needed
        capture.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        cv2.destroyAllWindows()