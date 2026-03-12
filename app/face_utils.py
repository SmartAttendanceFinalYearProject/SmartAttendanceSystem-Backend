import numpy as np
from insightface.app import FaceAnalysis
from typing import Optional, List
import cv2
from PIL import Image
import logging

logger = logging.getLogger(__name__)

face_app = FaceAnalysis(
    name='buffalo_l',
    providers=['CPUExecutionProvider']
)
face_app.prepare(ctx_id=0, det_size=(640, 640))  # or try (960, 960) / (1280, 1280) if small faces are an issue

def extract_face_embedding(image: Image.Image) -> Optional[np.ndarray]:
    """Detect faces + return embedding (used for registration)"""
    try:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        faces = face_app.get(img_cv)

        if not faces:
            logger.warning("No face detected")
            return None

        # Take the best face (highest confidence)
        best_face = max(faces, key=lambda f: f.det_score if f.det_score else 0)
        return best_face.normed_embedding.astype(np.float32)

    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return None


def detect_faces_for_attendance(image: Image.Image) -> List[dict]:
    """
    NEW FUNCTION: Returns ALL detected faces with boxes + landmarks + pose + emotion hints
    This is what you will use for group classroom photos (30-40 students)
    """
    try:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        faces = face_app.get(img_cv)

        results = []
        for face in faces:
            results.append({
                "bbox": face.bbox.astype(int).tolist(),          # [x1, y1, x2, y2]
                "confidence": float(face.det_score),
                "embedding": face.normed_embedding.tolist(),
                "landmarks": face.landmark.astype(int).tolist() if hasattr(face, 'landmark') else None,
                "pose": {
                    "yaw": float(face.pose[0]),
                    "pitch": float(face.pose[1]),
                    "roll": float(face.pose[2])
                } if hasattr(face, 'pose') else None,
                # Basic emotion can be derived later or added via another model
            })
        return results

    except Exception as e:
        logger.error(f"Group detection failed: {e}")
        return []