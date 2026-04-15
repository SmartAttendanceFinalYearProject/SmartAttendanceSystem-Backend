from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import base64
from io import BytesIO
from PIL import Image
import io   
import numpy as np
import logging
import torch
from bson.objectid import ObjectId
from datetime import datetime
from .recognizer import recognize_faces_in_classroom
from .emotion import predict_emotion

# ── Import project modules ────────────────────────────────
from .models import StudentCreate, StudentOut, FaceBox, DetectionResponse, DetectionRequest
from .database import students_collection
from .face_utils import extract_face_embedding, detect_faces_for_attendance

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
        students_collection.find_one(limit=1)  # test mongo connection
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


@app.post("/register", response_model=StudentOut)
async def register_student(
    fullName: str = Form(..., min_length=2),  # Changed from name
    studentID: str = Form(...),  # Changed from student_id, now required
    department: Optional[str] = Form(None),  # New field
    section: Optional[str] = Form(None),  # New field
    email: Optional[str] = Form(None),  # New field
    image: UploadFile = File(...)
):
    """
    Enroll a new student:
    - fullName (required)
    - studentID (required)
    - department (optional)
    - section (optional)
    - email (optional)
    - image file (must contain one clear face)
    """
    try:
        # Validate email format if provided
        if email:
            import re
            email_regex = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
            if not re.match(email_regex, email):
                raise HTTPException(400, "Invalid email format")

        # Read uploaded image
        contents = await image.read()
        pil_image = Image.open(BytesIO(contents)).convert("RGB")

        # Extract embedding with insightface
        embedding = extract_face_embedding(pil_image)
        if embedding is None:
            raise HTTPException(400, "No valid face detected in the uploaded image. Try a clearer photo.")

        # Prepare document for MongoDB with new attributes
        user_doc = {
            "fullName": fullName.strip(),
            "studentID": studentID.strip(),
            "department": department.strip() if department else None,
            "section": section.strip() if section else None,
            "email": email.strip() if email else None,
            "faceEmbedding": embedding.tolist(),  # numpy array → list[float]
            "registrationDate": datetime.utcnow()  # Changed from created_at/updated_at
        }

        # Check if studentID already exists
        existing_student = students_collection.find_one({"studentID": studentID.strip()})
        if existing_student:
            raise HTTPException(400, f"Student with ID {studentID} already exists")

        # Save to MongoDB
        result = students_collection.insert_one(user_doc)
        user_id = str(result.inserted_id)

        logger.info(f"Student registered: {fullName} (ID: {studentID})")

        return StudentOut(
            id=user_id,
            fullName=fullName,
            studentID=studentID,
            department=user_doc["department"],
            section=user_doc["section"],
            email=user_doc["email"],
            registrationDate=user_doc["registrationDate"]
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Registration failed: {e}", exc_info=True)
        raise HTTPException(500, f"Registration failed: {str(e)}")

@app.post("/attendance/recognize")
async def recognize_classroom_attendance(file: UploadFile = File(...)):
    """
    Full Smart Attendance: Student Recognition + Emotion + Pose
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Detect all faces
        faces = detect_faces_for_attendance(image)
        
        if not faces:
            return {"status": "success", "message": "No faces detected", "results": []}
        
        results = []
        for face in faces:
            emb = np.array(face["embedding"], dtype=np.float32)
            
            # 1. Student Recognition from Database
            student_recog = recognize_student(emb)
            
            # 2. Crop face for emotion & pose
            x1, y1, x2, y2 = face["bbox"]
            face_crop = image.crop((x1, y1, x2, y2))
            
            # 3. Predict Emotion
            emotion_result = predict_emotion(face_crop)
            
            # 4. Predict Pose
            pose_result = predict_pose(face_crop)
            
            results.append({
                "bbox": face["bbox"],
                "student_id": student_recog["student_id"],
                "full_name": student_recog["full_name"],
                "recognition_confidence": student_recog["confidence"],
                "recognized": student_recog["recognized"],
                "emotion": emotion_result["emotion"],
                "emotion_confidence": emotion_result["emotion_confidence"],
                "pose": pose_result["pose"],
                "pose_confidence": pose_result["pose_confidence"]
            })

        present = [r["full_name"] for r in results if r["recognized"]]

        return {
            "status": "success",
            "total_faces_detected": len(faces),
            "recognized_count": len(present),
            "results": results,
            "present_list": present
        }

    except Exception as e:
        logger.error(f"Attendance error: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
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