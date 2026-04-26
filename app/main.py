from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Response
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
from .pose import predict_pose

# ── Import project modules ────────────────────────────────
from .models import StudentCreate, StudentOut, FaceBox, DetectionResponse, DetectionRequest
from .database import students_collection , teachers_collection, subjects_collection, classes_collection
from .face_utils import extract_face_embedding, detect_faces_for_attendance

# ── YOLO imports (only when needed) ───────────────────────
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# ───Auth imports ───────────────────
from .auth import create_access_token, get_current_user, get_password_hash, verify_password
from datetime import timedelta
from .models import (
    StudentCreate, StudentOut, FaceBox, DetectionResponse, DetectionRequest,
    UserLogin, Token, TeacherCreateByAdmin, TokenData, ClassCreate, ClassOut,
    ClassUpdate, SubjectOut, TeacherOut, TeacherUpdate,
    SubjectCreate, SubjectUpdate
)

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
    fullName: str = Form(..., min_length=2),
    studentID: str = Form(...),
    department: Optional[str] = Form(None),
    section: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    batch: str = Form(...),
    class_year: str = Form(...),
    semester: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Enroll a new student:
    - fullName (required)
    - studentID (required)
    - department (optional)
    - section (optional)
    - email (optional)
    - batch (required)
    - class_year (required)
    - semester (required)
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
            "batch": batch.strip(),
            "class_year": class_year.strip(),
            "semester": semester.strip(),
            "faceEmbedding": embedding.tolist(),  # numpy array → list[float]
            "registrationDate": datetime.utcnow()
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
            batch=user_doc["batch"],
            class_year=user_doc["class_year"],
            semester=user_doc["semester"],
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
    Full Attendance: Recognition + Emotion + Pose (Sitting/Standing)
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        faces = detect_faces_for_attendance(image)
        
        if not faces:
            return {"status": "success", "message": "No faces detected", "results": []}

        results = []
        for face in faces:
            emb = np.array(face["embedding"], dtype=np.float32)
            
            # Student Recognition
            student_recog = recognize_faces_in_classroom([face])[0]
            
            # Crop face for emotion
            x1, y1, x2, y2 = face["bbox"]
            face_crop = image.crop((x1, y1, x2, y2))
            
            emotion_result = predict_emotion(face_crop)
            # pose_result = predict_pose(face_crop, face["bbox"], image.width, image.height)
            pose_result = predict_pose(face, image.width, image.height)
            
            results.append({
                "bbox": face["bbox"],
                "student_id": student_recog["student_id"],
                "full_name": student_recog["full_name"],
                "recognition_confidence": student_recog["recognition_confidence"],
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


@app.post("/login", response_model=Token)
async def login(form_data: UserLogin):
    """
    Single login for both Admin and Teacher
    """
    user = teachers_collection.find_one({"username": form_data.username})
    
    if not user or not verify_password(form_data.password, user.get("password", "")):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password"
        )

    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]}
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "role": user["role"],
        "full_name": user.get("full_name")
    }

# Admin can create new teacher
@app.post("/admin/create-teacher")
async def create_teacher(
    teacher: TeacherCreateByAdmin, 
    current_user: TokenData = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(403, detail="Only admin can create teachers")
    
    if teachers_collection.find_one({"username": teacher.username}):
        raise HTTPException(400, detail="Username already exists")
    
    teacher_doc = {
        "full_name": teacher.full_name,
        "subject_id": teacher.subject_id,
        "username": teacher.username,
        "password": get_password_hash(teacher.password),
        "role": "teacher",
        "created_at": datetime.utcnow()
    }
    
    result = teachers_collection.insert_one(teacher_doc)
    return {"message": "Teacher created successfully", "teacher_id": str(result.inserted_id)}

@app.get("/admin/teachers", response_model=List[TeacherOut])
async def get_all_teachers(current_user: TokenData = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(403, detail="Only admin can view teachers")

    teachers = []
    for teacher in teachers_collection.find({"role": "teacher"}):
        teachers.append(
            TeacherOut(
                id=str(teacher["_id"]),
                full_name=teacher["full_name"],
                subject_id=teacher["subject_id"],
                username=teacher["username"]
            )
        )
    return teachers

@app.put("/admin/teachers/{teacher_id}", response_model=TeacherOut)
async def update_teacher(
    teacher_id: str,
    teacher_data: TeacherUpdate,
    current_user: TokenData = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(403, detail="Only admin can update teachers")

    try:
        obj_id = ObjectId(teacher_id)
    except:
        raise HTTPException(400, detail="Invalid teacher ID format")

    update_data = {k: v for k, v in teacher_data.dict(exclude_unset=True).items() if v is not None}
    if not update_data:
        raise HTTPException(400, detail="No update data provided")

    if "username" in update_data:
        existing_user = teachers_collection.find_one({
            "username": update_data["username"],
            "_id": {"$ne": obj_id}
        })
        if existing_user:
            raise HTTPException(400, detail="Username already exists")

    if "password" in update_data:
        update_data["password"] = get_password_hash(update_data["password"])

    updated_teacher = teachers_collection.find_one_and_update(
        {"_id": obj_id, "role": "teacher"},
        {"$set": update_data},
        return_document=True
    )

    if not updated_teacher:
        raise HTTPException(404, detail="Teacher not found")

    return TeacherOut(
        id=str(updated_teacher["_id"]),
        full_name=updated_teacher["full_name"],
        subject_id=updated_teacher["subject_id"],
        username=updated_teacher["username"]
    )

@app.delete("/admin/teachers/{teacher_id}", status_code=204)
async def delete_teacher(
    teacher_id: str,
    current_user: TokenData = Depends(get_current_user)
):
    if current_user.role != "admin":
        raise HTTPException(403, detail="Only admin can delete teachers")

    try:
        obj_id = ObjectId(teacher_id)
    except:
        raise HTTPException(400, detail="Invalid teacher ID format")

    result = teachers_collection.delete_one({"_id": obj_id, "role": "teacher"})
    if result.deleted_count == 0:
        raise HTTPException(404, detail="Teacher not found")

    return Response(status_code=204)

@app.get("/subjects", response_model=List[SubjectOut])
async def get_all_subjects(current_user: TokenData = Depends(get_current_user)):
    """
    Returns a list of all subjects in the database.
    Accessible by both Admin and Teacher.
    """
    if current_user.role not in ["admin", "teacher"]:
        raise HTTPException(status_code=403, detail="Access forbidden: requires admin or teacher role")
        
    subjects = []
    for subject in subjects_collection.find():
        subjects.append(SubjectOut(
            id=str(subject["_id"]),
            subject_name=subject["subject_name"],
            subject_code=subject["subject_code"]
        ))
    return subjects

# ====================== ADMIN: SUBJECT MANAGEMENT ======================

@app.post("/admin/subjects", response_model=SubjectOut)
async def create_subject(
    subject_data: SubjectCreate,
    current_user: TokenData = Depends(get_current_user)
):
    """
    Admin creates a new subject.
    """
    if current_user.role != "admin":
        raise HTTPException(403, "Only admin can create subjects")

    if subjects_collection.find_one({"subject_code": subject_data.subject_code}):
        raise HTTPException(400, f"Subject with code '{subject_data.subject_code}' already exists")

    subject_doc = {
        "subject_name": subject_data.subject_name,
        "subject_code": subject_data.subject_code,
        "created_at": datetime.utcnow()
    }
    
    result = subjects_collection.insert_one(subject_doc)
    
    return SubjectOut(
        id=str(result.inserted_id),
        subject_name=subject_doc["subject_name"],
        subject_code=subject_doc["subject_code"]
    )


@app.put("/admin/subjects/{subject_id}", response_model=SubjectOut)
async def update_subject(
    subject_id: str,
    subject_data: SubjectUpdate,
    current_user: TokenData = Depends(get_current_user)
):
    """
    Admin updates an existing subject's name or code.
    """
    if current_user.role != "admin":
        raise HTTPException(403, "Only admin can update subjects")

    try:
        obj_id = ObjectId(subject_id)
    except:
        raise HTTPException(400, "Invalid subject ID format")

    update_data = {k: v for k, v in subject_data.dict(exclude_unset=True).items()}
    if not update_data:
        raise HTTPException(400, "No update data provided")

    # Check for duplicate subject_code if it's being changed
    if "subject_code" in update_data:
        existing = subjects_collection.find_one({
            "subject_code": update_data["subject_code"],
            "_id": {"$ne": obj_id}
        })
        if existing:
            raise HTTPException(400, f"Another subject with code '{update_data['subject_code']}' already exists")

    updated_subject = subjects_collection.find_one_and_update(
        {"_id": obj_id},
        {"$set": update_data},
        return_document=True
    )

    if not updated_subject:
        raise HTTPException(404, "Subject not found")

    return SubjectOut(
        id=str(updated_subject["_id"]),
        subject_name=updated_subject["subject_name"],
        subject_code=updated_subject["subject_code"]
    )


@app.delete("/admin/subjects/{subject_id}", status_code=204)
async def delete_subject(
    subject_id: str,
    current_user: TokenData = Depends(get_current_user)
):
    """
    Admin deletes a subject.
    """
    if current_user.role != "admin":
        raise HTTPException(403, "Only admin can delete subjects")

    try:
        obj_id = ObjectId(subject_id)
    except:
        raise HTTPException(400, "Invalid subject ID format")

    result = subjects_collection.delete_one({"_id": obj_id})

    if result.deleted_count == 0:
        raise HTTPException(404, "Subject not found")

    return Response(status_code=204)


# ====================== ADMIN: CLASS MANAGEMENT ======================

@app.post("/admin/classes", response_model=ClassOut)
async def create_class(
    class_data: ClassCreate,
    current_user: TokenData = Depends(get_current_user)
):
    """
    Admin creates a new class.
    """
    if current_user.role != "admin":
        raise HTTPException(403, detail="Only admins can create classes")

    # Find subject by subject_code or _id
    subject = subjects_collection.find_one({"_id": ObjectId(class_data.subject_id)})
    if not subject:
        raise HTTPException(404, detail=f"Subject not found with id: {class_data.subject_id}")

    # Find teacher by teacher_id
    teacher = teachers_collection.find_one({"_id": ObjectId(class_data.teacher_id)})
    if not teacher:
        raise HTTPException(404, detail=f"Teacher not found with id: {class_data.teacher_id}")

    # Create class document
    class_doc = {
        "class_name": class_data.class_name,
        "subject_id": str(subject["_id"]),
        "subject_code": subject["subject_code"],
        "subject_name": subject["subject_name"],
        "teacher_id": str(teacher["_id"]),
        "teacher_name": teacher["full_name"],
        "start_date": class_data.start_date,
        "end_date": class_data.end_date,
        "schedule": class_data.schedule.dict(),
        "students": class_data.students,
        "created_at": datetime.utcnow(),
        "created_by": current_user.username
    }

    result = classes_collection.insert_one(class_doc)
    
    return ClassOut(
        id=str(result.inserted_id),
        class_name=class_doc["class_name"],
        subject_id=class_doc["subject_id"],
        teacher_id=class_doc["teacher_id"],
        teacher_name=class_doc["teacher_name"],
        start_date=class_doc["start_date"],
        end_date=class_doc["end_date"],
        schedule=class_data.schedule,
        student_count=len(class_doc["students"]),
        students=class_doc["students"]
    )

@app.put("/admin/classes/{class_id}", response_model=ClassOut)
async def update_class(
    class_id: str,
    class_data: ClassUpdate,
    current_user: TokenData = Depends(get_current_user)
):
    """
    Admin updates an existing class.
    """
    if current_user.role != "admin":
        raise HTTPException(403, "Only admin can update classes")

    try:
        obj_id = ObjectId(class_id)
    except:
        raise HTTPException(400, "Invalid class ID format")

    update_data = {k: v for k, v in class_data.dict(exclude_unset=True).items() if v is not None}
    
    if "teacher_id" in update_data:
        teacher = teachers_collection.find_one({"_id": ObjectId(update_data["teacher_id"])})
        if not teacher:
            raise HTTPException(404, f"Teacher with id {update_data['teacher_id']} not found")
        update_data["teacher_name"] = teacher["full_name"]

    if "subject_id" in update_data:
        subject = subjects_collection.find_one({"_id": ObjectId(update_data["subject_id"])})
        if not subject:
            raise HTTPException(404, f"Subject with id {update_data['subject_id']} not found")
        update_data["subject_code"] = subject["subject_code"]
        update_data["subject_name"] = subject["subject_name"]

    if not update_data:
        raise HTTPException(400, "No update data provided")

    updated_class = classes_collection.find_one_and_update(
        {"_id": obj_id},
        {"$set": update_data},
        return_document=True
    )

    if not updated_class:
        raise HTTPException(404, "Class not found")

    return ClassOut(
        id=str(updated_class["_id"]),
        class_name=updated_class["class_name"],
        subject_id=updated_class["subject_id"],
        teacher_id=updated_class["teacher_id"],
        teacher_name=updated_class["teacher_name"],
        start_date=updated_class["start_date"],
        end_date=updated_class["end_date"],
        schedule=updated_class["schedule"],
        student_count=len(updated_class.get("students", [])),
        students=updated_class.get("students", [])
    )

@app.delete("/admin/classes/{class_id}", status_code=204)
async def delete_class(
    class_id: str,
    current_user: TokenData = Depends(get_current_user)
):
    """
    Admin deletes a class.
    """
    if current_user.role != "admin":
        raise HTTPException(403, "Only admin can delete classes")

    try:
        obj_id = ObjectId(class_id)
    except:
        raise HTTPException(400, "Invalid class ID format")

    result = classes_collection.delete_one({"_id": obj_id})

    if result.deleted_count == 0:
        raise HTTPException(404, "Class not found")

    return Response(status_code=204)

# ====================== ADMIN: STUDENT LIST ======================

@app.get("/admin/students")
async def get_all_students(current_user: TokenData = Depends(get_current_user)):
    """
    Returns a lightweight list of all registered students (id, fullName, studentID).
    Admin-only. Used to populate dropdowns on the frontend.
    """
    if current_user.role != "admin":
        raise HTTPException(403, detail="Only admin can view student list")

    students = []
    for s in students_collection.find({}, {"_id": 1, "fullName": 1, "studentID": 1, "batch": 1, "class_year": 1, "semester": 1, "section": 1, "department": 1}):
        students.append({
            "id": str(s["_id"]),
            "fullName": s.get("fullName", ""),
            "studentID": s.get("studentID", ""),
            "batch": s.get("batch", ""),
            "class_year": s.get("class_year", ""),
            "semester": s.get("semester", ""),
            "section": s.get("section", ""),
            "department": s.get("department", "")
        })
    return students


# ====================== GENERAL CLASS ENDPOINTS ======================

@app.get("/classes", response_model=List[ClassOut])
async def get_all_classes(current_user: TokenData = Depends(get_current_user)):
    """
    Get all classes. Accessible by both Admin and Teacher.
    """
    if current_user.role not in ["admin", "teacher"]:
        raise HTTPException(403, "Access forbidden")

    classes = []
    for cls in classes_collection.find():
        try:
            # Coerce the stored schedule dict into the Pydantic model
            raw_schedule = cls.get("schedule", {"schedule": []})
            if isinstance(raw_schedule, dict):
                from .models import ClassSchedule, DaySchedule as DS
                day_rows = raw_schedule.get("schedule", [])
                schedule_obj = ClassSchedule(
                    schedule=[DS(**d) if isinstance(d, dict) else d for d in day_rows]
                )
            else:
                schedule_obj = raw_schedule

            classes.append(ClassOut(
                id=str(cls["_id"]),
                class_name=cls.get("class_name", ""),
                subject_id=cls.get("subject_id", ""),
                teacher_id=cls.get("teacher_id", ""),
                teacher_name=cls.get("teacher_name", ""),
                start_date=cls["start_date"],
                end_date=cls["end_date"],
                schedule=schedule_obj,
                student_count=len(cls.get("students", [])),
                students=cls.get("students", [])
            ))
        except Exception as e:
            logger.warning(f"Skipping malformed class doc {cls.get('_id')}: {e}")
            continue
    return classes

@app.get("/classes/{class_id}", response_model=ClassOut)
async def get_class(class_id: str, current_user: TokenData = Depends(get_current_user)):
    """
    Get a single class by its ID. Accessible by both Admin and Teacher.
    """
    if current_user.role not in ["admin", "teacher"]:
        raise HTTPException(403, "Access forbidden")
        
    try:
        obj_id = ObjectId(class_id)
    except:
        raise HTTPException(400, "Invalid class ID format")

    cls = classes_collection.find_one({"_id": obj_id})
    if not cls:
        raise HTTPException(404, "Class not found")

    try:
        raw_schedule = cls.get("schedule", {"schedule": []})
        if isinstance(raw_schedule, dict):
            from .models import ClassSchedule, DaySchedule as DS
            day_rows = raw_schedule.get("schedule", [])
            schedule_obj = ClassSchedule(
                schedule=[DS(**d) if isinstance(d, dict) else d for d in day_rows]
            )
        else:
            schedule_obj = raw_schedule
    except Exception as e:
        logger.warning(f"Schedule parse error for class {class_id}: {e}")
        from .models import ClassSchedule
        schedule_obj = ClassSchedule(schedule=[])

    return ClassOut(
        id=str(cls["_id"]),
        class_name=cls.get("class_name", ""),
        subject_id=cls.get("subject_id", ""),
        teacher_id=cls.get("teacher_id", ""),
        teacher_name=cls.get("teacher_name", ""),
        start_date=cls["start_date"],
        end_date=cls["end_date"],
        schedule=schedule_obj,
        student_count=len(cls.get("students", [])),
        students=cls.get("students", [])
    )
        


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