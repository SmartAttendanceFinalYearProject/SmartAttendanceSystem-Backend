from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import logging
import sys
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Face Detection API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
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

class YOLOFaceDetector:
    """
    YOLO Face Detector using ultralytics
    """
    def __init__(self, model_name: str = "yolov8n.pt"):
        self.model = None
        self.model_name = model_name
        self.model_info = {}
        logger.info(f"Initializing YOLO Face Detector with model: {model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model"""
        try:
            from ultralytics import YOLO
            from ultralytics.nn.tasks import DetectionModel
            
            # Allowlist the required global for safe unpickling (PyTorch 2.1+ security)
            torch.serialization.add_safe_globals([DetectionModel])
            
            logger.info(f"Loading YOLO model: {self.model_name}")
            
            # Load the model
            self.model = YOLO(self.model_name)
            
            # Test with dummy image to ensure it's loaded
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            results = self.model(dummy_image, verbose=False)
            
            self.model_info = {
                "model_name": self.model_name,
                "model_type": "yolo",
                "classes": self.model.names if hasattr(self.model, 'names') else {},
                "input_size": 640
            }
            
            logger.info(f"✅ YOLO model loaded successfully: {self.model_name}")
            logger.info(f"Model classes: {self.model.names if hasattr(self.model, 'names') else 'Unknown'}")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
            raise Exception(f"YOLO model loading failed: {e}")
    
    def detect_faces(self, image: Image.Image) -> List[FaceBox]:
        """
        Detect faces in image using YOLO
        """
        try:
            if self.model is None:
                raise Exception("YOLO model not loaded")
            
            # Convert PIL Image to numpy array
            image_np = np.array(image)
            
            # Ensure image is in RGB format
            if len(image_np.shape) == 2:  # Grayscale
                image_np = np.stack([image_np] * 3, axis=-1)
            elif image_np.shape[2] == 4:  # RGBA
                image_np = image_np[:, :, :3]
            
            logger.info(f"Image shape: {image_np.shape}, dtype: {image_np.dtype}")
            
            # Run YOLO inference
            results = self.model(
                image_np,
                conf=0.25,      # Confidence threshold
                iou=0.45,       # NMS IoU threshold
                imgsz=640,      # Inference size
                verbose=False,
                device='cpu'    # change to 'cuda' if you have GPU
            )
            
            # Extract face detections
            faces = []
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    logger.info(f"Found {len(result.boxes)} detections")
                    
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        # Get class ID and name
                        class_id = int(box.cls[0].cpu().numpy()) if len(box.cls) > 0 else -1
                        class_name = self.model.names.get(class_id, f'class_{class_id}') if hasattr(self.model, 'names') else 'unknown'
                        
                        logger.info(f"  Detection: {class_name} (conf: {confidence:.3f}) at [{x1},{y1},{x2},{y2}]")
                        
                        # Filter for persons (class 0) or faces
                        if class_name.lower() in ['person', 'face'] or class_id == 0:
                            if confidence >= 0.25:
                                faces.append(FaceBox(
                                    x1=int(x1),
                                    y1=int(y1),
                                    x2=int(x2),
                                    y2=int(y2),
                                    confidence=round(confidence, 3)
                                ))
                else:
                    logger.info("No detections in this result")
            
            logger.info(f"✅ Total faces detected: {len(faces)}")
            return faces
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}", exc_info=True)
            return []

# Initialize detector
try:
    detector = YOLOFaceDetector(model_name="yolov8n.pt")
    logger.info("✅ Face detector initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize detector: {e}", exc_info=True)
    detector = None

def decode_base64_image(image_base64: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    try:
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        logger.info(f"Decoded image: {image.size} ({image.mode})")
        return image
        
    except Exception as e:
        logger.error(f"Image decoding error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

@app.get("/")
async def root():
    if detector:
        status = "ready"
        model_info = detector.model_info
    else:
        status = "not_ready"
        model_info = {"error": "Detector not initialized"}
    
    return {
        "message": "YOLO Face Detection API",
        "status": status,
        "endpoints": {
            "detect_faces": "POST /detect",
            "health": "GET /health",
            "test": "GET /test"
        },
        "model_info": model_info
    }

@app.get("/health")
async def health_check():
    if detector and detector.model:
        status = "healthy"
        model_status = "loaded"
    else:
        status = "unhealthy"
        model_status = "not_loaded"
    
    return {
        "status": status,
        "service": "face_detection",
        "yolo_model": model_status,
        "python_version": sys.version.split()[0]
    }

@app.get("/test")
async def test_detection():
    try:
        from PIL import Image, ImageDraw
        import io
        
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        positions = [(100, 100), (300, 150), (500, 80)]
        sizes = [80, 90, 70]
        
        for i, (x, y) in enumerate(positions):
            size = sizes[i]
            draw.ellipse(
                [x, y, x + size, y + size],
                fill='lightblue',
                outline='darkblue',
                width=3
            )
        
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        image = Image.open(io.BytesIO(base64.b64decode(img_str)))
        
        if detector:
            faces = detector.detect_faces(image)
            message = f"Test completed: {len(faces)} faces detected"
        else:
            faces = []
            message = "Detector not available"
        
        return {
            "total_faces": len(faces),
            "faces": [face.dict() for face in faces],
            "message": message,
            "test_image": "3 circles drawn"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")

@app.post("/detect", response_model=DetectionResponse)
async def detect_faces(request: DetectionRequest):
    try:
        logger.info("=" * 50)
        logger.info("Received face detection request")
        
        if detector is None or detector.model is None:
            raise HTTPException(
                status_code=500,
                detail="Face detector not initialized. Please check server logs."
            )
        
        image = decode_base64_image(request.image_base64)
        logger.info(f"Processing image: {image.size} pixels")
        
        faces = detector.detect_faces(image)
        
        if len(faces) == 0:
            message = "No faces detected"
        elif len(faces) == 1:
            message = "1 face detected"
        else:
            message = f"{len(faces)} faces detected"
        
        logger.info(f"✅ Detection complete: {message}")
        logger.info("=" * 50)
        
        return DetectionResponse(
            total_faces=len(faces),
            faces=faces,
            message=message,
            model_info=detector.model_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Face detection failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    logger.info("=" * 50)
    logger.info("Starting YOLO Face Detection API")
    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"Ultralytics: {getattr(detector.model if detector and detector.model else None, '__class__', 'N/A')}")
    logger.info("=" * 50)
    
    if detector is None:
        logger.error("❌ Detector failed to initialize. Check logs above.")
    else:
        logger.info("✅ Detector ready for requests")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )