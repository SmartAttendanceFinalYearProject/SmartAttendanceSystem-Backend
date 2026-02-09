from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import logging
import torch

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Face Detection API", version="1.0.0")

# CORS (you can restrict origins later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ────────────────────────────────────────────────

class DetectionRequest(BaseModel):
    image_base64: str

class FaceBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

class DetectionResponse(BaseModel):
    total_faces: int
    faces: List[FaceBox]
    message: str
    model_info: Optional[dict] = None

# ── YOLO Face Detector ────────────────────────────────────

class YOLOFaceDetector:
    def __init__(self, model_name: str = "yolov8n.pt"):
        self.model = None
        self.model_name = model_name
        self.model_info = {}
        logger.info(f"Loading model: {model_name}")
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            from ultralytics.nn.tasks import DetectionModel

            # Required for PyTorch >= 2.1 when loading some older pickled models
            torch.serialization.add_safe_globals([DetectionModel])

            self.model = YOLO(self.model_name)

            self.model_info = {
                "model_name": self.model_name,
                "model_type": "yolo",
                "classes": self.model.names if hasattr(self.model, 'names') else {},
                "input_size": 640
            }

            logger.info(f"Model loaded successfully: {self.model_name}")

        except Exception as e:
            logger.error(f"Model loading failed: {e}", exc_info=True)
            raise

    def detect_faces(self, image: Image.Image) -> List[FaceBox]:
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded")

            image_np = np.array(image.convert("RGB"))

            results = self.model(
                image_np,
                conf=0.25,
                iou=0.45,
                imgsz=640,
                verbose=False,
                device="cpu"          # ← change to "cuda" if GPU available
            )

            faces = []

            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf)
                    cls_id = int(box.cls) if len(box.cls) > 0 else -1
                    cls_name = self.model.names.get(cls_id, "unknown")

                    # Accept person (class 0) or any face class
                    if cls_name.lower() in {"person", "face"} or cls_id == 0:
                        if conf >= 0.25:
                            faces.append(FaceBox(
                                x1=x1, y1=y1, x2=x2, y2=y2,
                                confidence=round(conf, 3)
                            ))

            return faces

        except Exception as e:
            logger.error(f"Detection error: {e}", exc_info=True)
            return []

# Global detector instance
try:
    detector = YOLOFaceDetector(model_name="yolov8n.pt")
    logger.info("Face detector ready")
except Exception as e:
    logger.critical(f"Failed to initialize detector: {e}", exc_info=True)
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
    if detector:
        return {
            "message": "YOLO Face Detection API",
            "status": "ready",
            "endpoints": {
                "detect": "POST /detect",
                "health": "GET /health"
            },
            "model_info": detector.model_info
        }
    return {"status": "not_ready", "error": "Detector not initialized"}

@app.get("/health")
async def health_check():
    if detector and detector.model:
        return {"status": "healthy", "model": "loaded"}
    return {"status": "unhealthy", "model": "not_loaded"}

@app.post("/detect", response_model=DetectionResponse)
async def detect_faces(request: DetectionRequest):
    if detector is None or detector.model is None:
        raise HTTPException(500, "Face detector not initialized")

    try:
        image = decode_base64_image(request.image_base64)
        faces = detector.detect_faces(image)

        count = len(faces)
        if count == 0:
            msg = "No faces detected"
        elif count == 1:
            msg = "1 face detected"
        else:
            msg = f"{count} faces detected"

        return DetectionResponse(
            total_faces=count,
            faces=faces,
            message=msg,
            model_info=detector.model_info
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        raise HTTPException(500, f"Detection failed: {str(e)}")

# ── Run ───────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting server...")
    logger.info(f"PyTorch: {torch.__version__}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,           # ← recommended during development
        log_level="info"
    )