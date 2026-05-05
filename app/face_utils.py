import numpy as np
from insightface.app import FaceAnalysis
from typing import Optional, List
import cv2
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# Load InsightFace once
face_app = FaceAnalysis(
    name='buffalo_l',
    providers=['CPUExecutionProvider']
)
face_app.prepare(ctx_id=0, det_size=(640, 640))

def extract_face_embedding(image: Image.Image) -> Optional[np.ndarray]:
    """Used only during student registration"""
    try:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        faces = face_app.get(img_cv)

        if not faces:
            logger.warning("No face detected")
            return None

        best_face = max(faces, key=lambda f: f.det_score if f.det_score is not None else 0)
        return best_face.normed_embedding.astype(np.float32)

    except Exception as e:
        logger.error(f"Embedding failed: {e}", exc_info=True)
        return None


def detect_faces_for_attendance(image: Image.Image) -> List[dict]:
    """
    Returns ALL detected faces with bbox + embedding (for group attendance)
    """
    try:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        faces = face_app.get(img_cv)

        results = []
        for face in faces:
            landmarks = None
            if hasattr(face, 'landmark') and face.landmark is not None:
                landmarks = face.landmark.astype(int).tolist()

            pose = None
            if hasattr(face, 'pose') and face.pose is not None:
                pose = {
                    "yaw": float(face.pose[0]),
                    "pitch": float(face.pose[1]),
                    "roll": float(face.pose[2])
                }

            results.append({
                "bbox": face.bbox.astype(int).tolist(),   # [x1, y1, x2, y2]
                "confidence": float(face.det_score or 0.0),
                "embedding": face.normed_embedding.tolist(),
                "landmarks": landmarks,
                "pose": pose,
            })
        return results

    except Exception as e:
        logger.error(f"Group detection failed: {e}", exc_info=True)
        return []