# funcs.py
import logging
import os
import shutil
from datetime import datetime
import numpy as np
from numpy.linalg import norm


def setup_logger(name, log_file, level=logging.DEBUG):
    """Setup and create logger with given parameters."""
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)  # Set to DEBUG to capture detailed logs

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


os.makedirs('logs', exist_ok=True)
logger = setup_logger("flog", "logs/flog.log")


def extract_date_from_filename(filename):
    """Extract date from filename."""
    try:
        # Adjust the split index and format based on your filename pattern
        date_str = filename.split("_")[2]
        return datetime.strptime(date_str, "%Y%m%d%H%M%S%f")
    except Exception as e:
        logger.error(f"Error extracting date from filename: {e}")
        return None


def copy_files(file1, file2, dirname):
    """Copy files from one directory to another."""
    try:
        os.makedirs(dirname, exist_ok=True)
        shutil.copy(file1, os.path.join(dirname, os.path.basename(file1)))
    except Exception as e:
        logger.error(f"Error copying files: {e}")


def get_faces_data(faces, min_confidence):
    """Return face data with maximum rectangle area."""
    if not faces:
        return None
    # Filter out faces with low confidence scores
    faces = [face for face in faces if face.det_score >= min_confidence]
    if not faces:
        return None
    # Return the face with the largest bounding box area
    return max(faces, key=lambda face: calculate_rectangle_area(face.bbox))


def calculate_rectangle_area(bbox):
    """Calculate rectangle area."""
    if len(bbox) != 4:
        raise ValueError("bbox must contain four coordinates: x_min, y_min, x_max, y_max")
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def compute_sim(feat1, feat2, logger=logger):
    """Compute similarity between two feature vectors."""
    try:
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        logger.debug(f"compute_sim: feat1 shape: {feat1.shape}, feat2 shape: {feat2.shape}")
        if feat1.shape != (512,) or feat2.shape != (512,):
            logger.error(f"Embeddings have incorrect shapes: feat1.shape={feat1.shape}, feat2.shape={feat2.shape}")
            return None
        sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
        return sim
    except Exception as e:
        logger.error(e)
        return None
