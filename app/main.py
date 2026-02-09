from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import logging
import torch
from bson.objectid import ObjectId
from datetime import datetime

# ── Import project modules ────────────────────────────────
from .models import UserCreate, UserOut, FaceBox, DetectionResponse, DetectionRequest
from .database import users_collection
from .face_utils import extract_face_embedding

# ── YOLO imports (only when needed) ───────────────────────
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart Attendance - Detection + Registration",
    description="Face detection (YOLOv8) + Student enrollment with embedding storage",
    version="0.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global YOLO Detector ──────────────────────────────────

class YOLOFaceDetector:
    def __init__(self, model_name: str = "yolov8n.pt"):
        self.model = None
        self.model_name = model_name
        self.model_info = {}
        logger.info(f"Loading YOLO model: {model_name}")
        self._load_model()

    def _load_model(self):
        try:
            torch.serialization.add_safe_globals([DetectionModel])
            self.model = YOLO(self.model_name)

            self.model_info = {
                "model_name": self.model_name,
                "model_type": "yolo",
                "classes": self.model.names if hasattr(self.model, 'names') else {},
                "input_size": 640
            }
            logger.info(f"YOLO model loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"YOLO load failed: {e}", exc_info=True)
            raise

    def detect_faces(self, image: Image.Image) -> List[FaceBox]:
        try:
            if self.model is None:
                raise RuntimeError("YOLO model not loaded")

            image_np = np.array(image.convert("RGB"))

            results = self.model(
                image_np,
                conf=0.25,
                iou=0.45,
                imgsz=640,
                verbose=False,
                device="cpu"          # change to "cuda" if you have GPU
            )

            faces = []

            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf)
                    cls_id = int(box.cls) if len(box.cls) > 0 else -1
                    cls_name = self.model.names.get(cls_id, "unknown")

                    if cls_name.lower() in {"person", "face"} or cls_id == 0:
                        if conf >= 0.25:
                            faces.append(FaceBox(
                                x1=x1, y1=y1, x2=x2, y2=y2,
                                confidence=round(conf, 3)
                            ))

            return faces

        except Exception as e:
            logger.error(f"YOLO detection error: {e}", exc_info=True)
            return []


# Initialize YOLO detector
detector = None
try:
    detector = YOLOFaceDetector(model_name="yolov8n.pt")
    logger.info("YOLO face detector ready")
except Exception as e:
    logger.critical(f"Failed to initialize YOLO detector: {e}", exc_info=True)
    detector = None


# ── Helpers ───────────────────────────────────────────────

def decode_base64_image(image_base64: str) -> Image.Image:
    try:
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        data = base64.b64decode(image_base64)
        img = Image.open(BytesIO(data))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(400, detail=f"Invalid image data: {str(e)}")


# ── Endpoints ─────────────────────────────────────────────

@app.get("/")
async def root():
    status = "ready" if detector else "not_ready"
    return {
        "message": "Smart Attendance API",
        "status": status,
        "endpoints": {
            "detect": "POST /detect (face detection)",
            "register": "POST /register (student enrollment)",
            "health": "GET /health"
        }
    }


@app.get("/health")
async def health_check():
    yolo_ok = bool(detector and detector.model)
    try:
        users_collection.find_one(limit=1)  # test mongo connection
        db_ok = True
    except:
        db_ok = False

    return {
        "status": "healthy" if yolo_ok and db_ok else "partial",
        "yolo": "loaded" if yolo_ok else "failed",
        "mongodb": "connected" if db_ok else "disconnected",
        "face_embedding_model": "insightface-buffalo_l"
    }


@app.post("/detect", response_model=DetectionResponse)
async def detect_faces(request: DetectionRequest):
    if detector is None:
        raise HTTPException(500, "Face detector not initialized")

    try:
        image = decode_base64_image(request.image_base64)
        faces = detector.detect_faces(image)

        count = len(faces)
        msg = "No faces detected" if count == 0 else \
              "1 face detected" if count == 1 else \
              f"{count} faces detected"

        return DetectionResponse(
            total_faces=count,
            faces=faces,
            message=msg,
            model_info=detector.model_info
        )

    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        raise HTTPException(500, f"Detection failed: {str(e)}")


@app.post("/register", response_model=UserOut)
async def register_student(
    name: str = Form(..., min_length=2),
    student_id: Optional[str] = Form(None),
    group: Optional[str] = Form(None),
    image: UploadFile = File(...)
):
    """
    Enroll a new student:
    - name (required)
    - student_id (optional)
    - group/class (optional)
    - image file (must contain one clear face)
    """
    try:
        # Read uploaded image
        contents = await image.read()
        pil_image = Image.open(BytesIO(contents)).convert("RGB")

        # Extract embedding with insightface
        embedding = extract_face_embedding(pil_image)
        if embedding is None:
            raise HTTPException(400, "No valid face detected in the uploaded image. Try a clearer photo.")

        # Prepare document for MongoDB
        user_doc = {
            "name": name.strip(),
            "student_id": student_id.strip() if student_id else None,
            "group": group.strip() if group else None,
            "embedding": embedding.tolist(),  # numpy array → list[float]
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        # Save to MongoDB
        result = users_collection.insert_one(user_doc)
        user_id = str(result.inserted_id)

        logger.info(f"Student registered: {name} (ID: {user_id})")

        return UserOut(
            id=user_id,
            name=name,
            student_id=user_doc["student_id"],
            group=user_doc["group"]
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Registration failed: {e}", exc_info=True)
        raise HTTPException(500, f"Registration failed: {str(e)}")


# ── Run ───────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Smart Attendance API...")
    logger.info(f"PyTorch version: {torch.__version__}")

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,           # good for development
        log_level="info"
    )