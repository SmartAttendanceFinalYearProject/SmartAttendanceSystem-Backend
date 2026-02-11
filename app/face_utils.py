import numpy as np
from insightface.app import FaceAnalysis
from typing import Optional
import cv2
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# ── Global InsightFace analyzer ───────────────────────────
# Loaded once when module is imported
face_app = FaceAnalysis(
    name='buffalo_l',                       # 512-dim, good balance speed/accuracy in 2025–2026
    providers=['CPUExecutionProvider']      # change to 'CUDAExecutionProvider' if you have NVIDIA GPU + onnxruntime-gpu
)
face_app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 → CPU; -1 → GPU if available


def extract_face_embedding(image: Image.Image) -> Optional[np.ndarray]:
    """
    Detect the largest/most confident face in the image and return its
    normalized 512-dimensional embedding (vector).
    
    Returns None if:
    - No face detected
    - Multiple faces (we take only the best one)
    - Any processing error
    """
    try:
        # Convert PIL → OpenCV BGR (insightface expects BGR)
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Run detection + embedding extraction
        faces = face_app.get(img_cv)

        if len(faces) == 0:
            logger.warning("No face detected in the provided image")
            return None

        # Select the face with highest detection confidence
        best_face = max(faces, key=lambda f: f.det_score if f.det_score is not None else 0.0)

        # Return L2-normalized embedding (512 floats)
        return best_face.normed_embedding.astype(np.float32)

    except Exception as e:
        logger.error(f"Face embedding extraction failed: {str(e)}", exc_info=True)
        return None