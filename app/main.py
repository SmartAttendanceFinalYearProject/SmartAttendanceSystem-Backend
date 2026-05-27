from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import base64
import json
from io import BytesIO
from PIL import Image
import io
import cv2
import asyncio
import numpy as np
import logging
import torch
from bson.objectid import ObjectId
from datetime import datetime
from .recog_emotion_predict import predict_recog_emotion
from .pose_predict import predict_pose

# ── Import project modules ────────────────────────────────
from .models import StudentCreate, StudentOut, StudentMinimal, FaceBox, DetectionResponse, DetectionRequest
from .database import students_collection , teachers_collection, subjects_collection, classes_collection
from .face_utils import extract_face_embedding, detect_faces_for_attendance

# This module is the FastAPI application entrypoint for the Smart Attendance backend.
# It initializes the face detection model, connects to MongoDB collections,
# and exposes endpoints for detection, registration, attendance, authentication,
# teacher/admin management, and live streaming via WebSocket.

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
    SubjectCreate, SubjectUpdate, AttendanceRecord, AttendanceSession
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
# This class encapsulates YOLO face detection so the model is loaded once
# and reused for every incoming request.
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
    # Convert a base64-encoded image string into a PIL RGB image.
    # This supports both raw base64 content and data URLs like data:image/jpeg;base64,...
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
    # Basic API root endpoint used to verify service status.
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
    # Detect faces in a posted base64 image and return bounding boxes.
    # This endpoint does not do recognition, only face localization.
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


# Endpoint to register a student by uploading a photo, extracting a face embedding, and saving profile data.
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
        # Validate the email format if provided
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

# Endpoint to recognize student attendance from a classroom image and return emotion and pose data.
@app.post("/attendance/recognize")
async def recognize_classroom_attendance(file: UploadFile = File(...)):
    # Process a full classroom image, detect faces, recognize students, predict emotion,
    # and estimate pose for each detected face.
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Get faces with InsightFace (full context)
        faces = detect_faces_for_attendance(image)
        
        if not faces:
            return {
                "status": "success", 
                "message": "No faces detected", 
                "total_faces_detected": 0,
                "results": []
            }

        results = []
        image_width = image.width
        image_height = image.height

        for face in faces:
            # Crop only for recognition + emotion
            x1, y1, x2, y2 = face["bbox"]
            face_crop = image.crop((x1, y1, x2, y2))
            
            # Recognition + Emotion, passing the pre-computed embedding
            recog_emotion = predict_recog_emotion(face_crop, embedding=face.get("embedding"))
            
            # Pose - Use full image context with the face dictionary
            pose_result = predict_pose(face, image_width, image_height)
            
            results.append({
                "bbox": face["bbox"],
                "student_id": recog_emotion["student_id"],
                "full_name": recog_emotion["full_name"],
                "emotion": recog_emotion["emotion"],
                "pose": pose_result["pose"],
                "recognized": recog_emotion["recognized"],
                "pose_confidence": pose_result.get("pose_confidence", 60.0)
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
        logger.error(f"Recognition error: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

# Endpoint to finalize attendance records for a class session and save them to MongoDB.
@app.post("/attendance/approve")
async def approve_attendance(
    class_id: str = Form(...),
    session_date: str = Form(...),
    start_time: str = Form(...),
    end_time: str = Form(...),
    records_json: str = Form(...), # JSON string of AttendanceRecord list
    current_user: TokenData = Depends(get_current_user)
):
    """
    Finalize and save attendance records for a specific session.
    """
    try:
        import json
        records_data = json.loads(records_json)
        
        try:
            obj_id = ObjectId(class_id)
        except:
            raise HTTPException(400, "Invalid class ID format")
            
        cls = classes_collection.find_one({"_id": obj_id})
        if not cls:
            raise HTTPException(404, "Class not found")

        # Parse session_date
        try:
            parsed_session_date = datetime.fromisoformat(session_date.replace('Z', '+00:00'))
        except:
            parsed_session_date = datetime.utcnow()

        # Handle absent students
        # The 'students' field in Class stores studentIDs (strings)
        class_students_ids = cls.get("students", [])
        recognized_student_ids = {r["student_id"] for r in records_data if r.get("status") == "present"}
        
        final_records = []
        for r in records_data:
            final_records.append(AttendanceRecord(**r))
            
        for s_id in class_students_ids:
            if s_id not in recognized_student_ids:
                # Check if already in final_records (maybe as unknown or already marked absent)
                if not any(r.student_id == s_id for r in final_records):
                    # Find student name
                    student_doc = students_collection.find_one({"studentID": s_id})
                    full_name = student_doc["fullName"] if student_doc else "Unknown Student"
                    
                    final_records.append(AttendanceRecord(
                        student_id=s_id,
                        full_name=full_name,
                        status="absent",
                        timestamp=datetime.utcnow()
                    ))

        # Check if session already exists (matching by date and times)
        existing_sessions = cls.get("attendance_sessions", [])
        target_session_index = -1
        
        for i, s in enumerate(existing_sessions):
            try:
                s_date_val = s.get("session_date")
                if isinstance(s_date_val, str):
                    s_date = datetime.fromisoformat(s_date_val.replace('Z', '+00:00'))
                else:
                    s_date = s_date_val
                
                # Match by date and optionally start/end time
                if s_date.date() == parsed_session_date.date() and \
                   s.get("start_time") == start_time and \
                   s.get("end_time") == end_time:
                    target_session_index = i
                    break
            except:
                pass
        
        if target_session_index >= 0:
            # Update existing session
            classes_collection.update_one(
                {"_id": obj_id},
                {"$set": {f"attendance_sessions.{target_session_index}.records": [r.dict() for r in final_records]}}
            )
            session_id = existing_sessions[target_session_index]["id"]
        else:
            # Create new session
            new_session = AttendanceSession(
                session_date=parsed_session_date,
                start_time=start_time,
                end_time=end_time,
                records=final_records
            )
            classes_collection.update_one(
                {"_id": obj_id},
                {"$push": {"attendance_sessions": new_session.dict()}}
            )
            session_id = new_session.id

        return {
            "status": "success",
            "message": "Attendance approved and saved",
            "session_id": session_id
        }

    except Exception as e:
        logger.error(f"Approval error: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


# Endpoint for teachers to retrieve the classes assigned to the logged-in teacher.
@app.get("/teacher/classes", response_model=List[ClassOut])
async def get_teacher_classes(current_user: TokenData = Depends(get_current_user)):
    """
    Get classes assigned to the logged-in teacher.
    """
    if current_user.role != "teacher":
        raise HTTPException(403, "Only teachers can access this endpoint")

    # Find the teacher document to get their ID if needed, 
    # but we can also search classes by teacher_name or username if stored.
    # Looking at create_class, teacher_id is stored as a string.
    
    teacher = teachers_collection.find_one({"username": current_user.username})
    if not teacher:
        raise HTTPException(404, "Teacher not found")
        
    teacher_id = str(teacher["_id"])
    
    classes = []
    for cls in classes_collection.find({"teacher_id": teacher_id}):
        try:
            # Parse schedule
            raw_schedule = cls.get("schedule", {"schedule": []})
            if isinstance(raw_schedule, dict):
                from .models import ClassSchedule, DaySchedule as DS
                day_rows = raw_schedule.get("schedule", [])
                schedule_obj = ClassSchedule(
                    schedule=[DS(**d) if isinstance(d, dict) else d for d in day_rows]
                )
            else:
                schedule_obj = raw_schedule

            # Parse attendance sessions
            sessions = []
            for s in cls.get("attendance_sessions", []):
                sessions.append(AttendanceSession(**s))

            # Populate student details
            student_details = []
            for s_id in cls.get("students", []):
                s_doc = students_collection.find_one({"studentID": s_id})
                if s_doc:
                    student_details.append(StudentMinimal(
                        id=str(s_doc["_id"]),
                        fullName=s_doc["fullName"],
                        studentID=s_doc["studentID"]
                    ))

            classes.append(ClassOut(
                id=str(cls["_id"]),
                class_name=cls.get("class_name", ""),
                subject_id=cls.get("subject_id", ""),
                subject_name=cls.get("subject_name", ""),
                subject_code=cls.get("subject_code", ""),
                teacher_id=cls.get("teacher_id", ""),
                teacher_name=cls.get("teacher_name", ""),
                start_date=cls["start_date"],
                end_date=cls["end_date"],
                schedule=schedule_obj,
                student_count=len(cls.get("students", [])),
                students=cls.get("students", []),
                student_details=student_details,
                attendance_sessions=sessions
            ))
        except Exception as e:
            logger.warning(f"Skipping malformed class doc {cls.get('_id')}: {e}")
            continue
    return classes


# Authentication endpoint for teacher/admin login that returns a JWT bearer token.
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

# Admin-only endpoint to create a new teacher account and store hashed password.
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

# Admin endpoint to list all teachers in the system.
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

# Admin endpoint to update an existing teacher's details, including password if provided.
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

# Admin endpoint to delete a teacher account from the database.
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

# Endpoint to list all subjects. Accessible to both admins and teachers.
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

# ====================== ADMIN: SUBJECT MANAGEMENT ==================

# Admin endpoint to create a new subject record.
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


# Admin endpoint to update subject name or code.
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


# Admin endpoint to delete a subject from the catalog.
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

# Admin endpoint to create a class, assign it to a teacher, and attach students.
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
        subject_name=class_doc["subject_name"],
        subject_code=class_doc["subject_code"],
        teacher_id=class_doc["teacher_id"],
        teacher_name=class_doc["teacher_name"],
        start_date=class_doc["start_date"],
        end_date=class_doc["end_date"],
        schedule=class_data.schedule,
        student_count=len(class_doc["students"]),
        students=class_doc["students"]
    )

# Admin endpoint to update class details, teacher assignment, or subject association.
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

# Admin endpoint to delete a class and all its schedule/attendance metadata.
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

# Admin endpoint to list all registered students in a lightweight format.
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

# Endpoint for admin or teacher to list all classes with schedules, student details, and attendance sessions.
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

            # Parse attendance sessions
            sessions = []
            for s in cls.get("attendance_sessions", []):
                sessions.append(AttendanceSession(**s))

            # Populate student details
            student_details = []
            for s_id in cls.get("students", []):
                s_doc = students_collection.find_one({"studentID": s_id})
                if s_doc:
                    student_details.append(StudentMinimal(
                        id=str(s_doc["_id"]),
                        fullName=s_doc["fullName"],
                        studentID=s_doc["studentID"]
                    ))

            classes.append(ClassOut(
                id=str(cls["_id"]),
                class_name=cls.get("class_name", ""),
                subject_id=cls.get("subject_id", ""),
                subject_name=cls.get("subject_name", ""),
                subject_code=cls.get("subject_code", ""),
                teacher_id=cls.get("teacher_id", ""),
                teacher_name=cls.get("teacher_name", ""),
                start_date=cls["start_date"],
                end_date=cls["end_date"],
                schedule=schedule_obj,
                student_count=len(cls.get("students", [])),
                students=cls.get("students", []),
                student_details=student_details,
                attendance_sessions=sessions
            ))
        except Exception as e:
            logger.warning(f"Skipping malformed class doc {cls.get('_id')}: {e}")
            continue
    return classes

# Endpoint for admin or teacher to retrieve a single class by ID, including its students and attendance sessions.
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

    # Parse attendance sessions
    sessions = []
    for s in cls.get("attendance_sessions", []):
        sessions.append(AttendanceSession(**s))

    # Populate student details
    student_details = []
    for s_id in cls.get("students", []):
        s_doc = students_collection.find_one({"studentID": s_id})
        if s_doc:
            student_details.append(StudentMinimal(
                id=str(s_doc["_id"]),
                fullName=s_doc["fullName"],
                studentID=s_doc["studentID"]
            ))

    return ClassOut(
        id=str(cls["_id"]),
        class_name=cls.get("class_name", ""),
        subject_id=cls.get("subject_id", ""),
        subject_name=cls.get("subject_name", ""),
        subject_code=cls.get("subject_code", ""),
        teacher_id=cls.get("teacher_id", ""),
        teacher_name=cls.get("teacher_name", ""),
        start_date=cls["start_date"],
        end_date=cls["end_date"],
        schedule=schedule_obj,
        student_count=len(cls.get("students", [])),
        students=cls.get("students", []),
        student_details=student_details,
        attendance_sessions=sessions
    )

# Admin dashboard endpoint that computes school analytics such as average attendance and department breakdowns.
@app.get("/admin/analytics/stats")
async def get_analytics_stats(current_user: TokenData = Depends(get_current_user)):
    """
    Get aggregated analytics data for the admin dashboard.
    """
    if current_user.role != "admin":
        raise HTTPException(403, detail="Only admin can access analytics")

    total_students = students_collection.count_documents({})
    total_teachers = teachers_collection.count_documents({"role": "teacher"})
    total_classes = classes_collection.count_documents({})

    # Calculate average attendance across all sessions
    total_present = 0
    total_records = 0
    
    # For weekly trend (last 7 days)
    today = datetime.utcnow().date()
    weekly_trend = []
    # Initialize last 7 days including today
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        weekly_trend.append({
            "date": day,
            "day": day.strftime("%a"), 
            "present": 0, 
            "total": 0
        })

    # Department breakdown
    pipeline = [
        {"$group": {"_id": "$department", "count": {"$sum": 1}}},
        {"$project": {"name": {"$ifNull": ["$_id", "Unspecified"]}, "count": 1, "_id": 0}}
    ]
    departments = list(students_collection.aggregate(pipeline))

    # Iterate classes to get attendance stats
    for cls in classes_collection.find():
        for session in cls.get("attendance_sessions", []):
            s_date_val = session.get("session_date")
            if isinstance(s_date_val, str):
                s_date = datetime.fromisoformat(s_date_val.replace('Z', '+00:00'))
            else:
                s_date = s_date_val
            
            s_records = session.get("records", [])
            p_count = sum(1 for r in s_records if r.get("status") == "present")
            t_count = len(s_records)
            
            total_present += p_count
            total_records += t_count
            
            # Check if it falls in our weekly trend range
            session_day = s_date.date()
            for entry in weekly_trend:
                if entry["date"] == session_day:
                    entry["present"] += p_count
                    entry["total"] += t_count
                    break

    avg_attendance = (total_present / total_records * 100) if total_records > 0 else 0
    
    # Format weekly data for frontend
    formatted_weekly = []
    for entry in weekly_trend:
        rate = (entry["present"] / entry["total"] * 100) if entry["total"] > 0 else 0
        formatted_weekly.append({"day": entry["day"], "rate": round(rate, 1)})

    # Sort departments by count descending
    departments.sort(key=lambda x: x["count"], reverse=True)

    return {
        "stats": [
            {
                "title": "Average Attendance",
                "value": f"{round(avg_attendance, 1)}%",
                "change": "+2.1%", # Hardcoded for now as we don't have historical comparison
                "isPositive": True
            },
            {
                "title": "Total Students",
                "value": f"{total_students:,}",
                "change": f"+{total_students}", # Total since start
                "isPositive": True
            },
            {
                "title": "Total Classes",
                "value": str(total_classes),
                "change": "Active",
                "isPositive": True
            },
            {
                "title": "Active Teachers",
                "value": str(total_teachers),
                "change": "Verified",
                "isPositive": True
            }
        ],
        "weeklyData": formatted_weekly,
        "departments": departments,
        "totalStudentsRaw": total_students
    }
    
# ====================== ADMIN: STUDENT PERFORMANCE ======================

@app.get("/admin/students/performance")
async def get_students_performance(current_user: TokenData = Depends(get_current_user)):
    """
    Get detailed attendance performance for all students (Admin only)
    """
    if current_user.role != "admin":
        raise HTTPException(403, "Only admin can access student performance")

    students_perf = []

    for student_doc in students_collection.find():
        student_id = student_doc["studentID"]
        full_name = student_doc["fullName"]

        total_present = 0
        total_sessions = 0
        class_performances = []

        # Find all classes this student is enrolled in
        for cls in classes_collection.find({"students": student_id}):
            class_name = cls.get("class_name")
            subject_name = cls.get("subject_name", "Unknown")

            present_count = 0
            session_count = 0

            for session in cls.get("attendance_sessions", []):
                session_count += 1
                for record in session.get("records", []):
                    if record.get("student_id") == student_id and record.get("status") == "present":
                        present_count += 1
                        total_present += 1

            total_sessions += session_count
            if session_count > 0:
                rate = (present_count / session_count) * 100
                class_performances.append({
                    "class_id": str(cls["_id"]),
                    "class_name": class_name,
                    "subject_name": subject_name,
                    "attendance_rate": round(rate, 1),
                    "total_sessions": session_count,
                    "present_count": present_count
                })

        overall_rate = (total_present / total_sessions * 100) if total_sessions > 0 else 0

        students_perf.append({
            "student_id": str(student_doc["_id"]),
            "full_name": full_name,
            "studentID": student_id,
            "department": student_doc.get("department"),
            "batch": student_doc.get("batch"),
            "class_year": student_doc.get("class_year"),
            "semester": student_doc.get("semester"),
            "overall_attendance": round(overall_rate, 1),
            "total_sessions": total_sessions,
            "classes": class_performances
        })

    # Sort by overall attendance descending
    students_perf.sort(key=lambda x: x["overall_attendance"], reverse=True)
    return students_perf


@app.get("/admin/students/{student_id}/details")
async def get_student_details(student_id: str, current_user: TokenData = Depends(get_current_user)):
    """
    Get detailed info for a specific student including full attendance history
    """
    if current_user.role != "admin":
        raise HTTPException(403, "Only admin can access this")

    try:
        # Find student
        student = students_collection.find_one({"studentID": student_id})
        if not student:
            raise HTTPException(404, "Student not found")

        # Get all attendance records across classes
        attendance_history = []

        for cls in classes_collection.find({"students": student_id}):
            for session in cls.get("attendance_sessions", []):
                for record in session.get("records", []):
                    if record.get("student_id") == student_id:
                        attendance_history.append({
                            "class_name": cls.get("class_name"),
                            "subject_name": cls.get("subject_name"),
                            "session_date": session.get("session_date"),
                            "status": record.get("status"),
                            "emotion": record.get("emotion"),
                            "pose": record.get("pose")
                        })

        return {
            "student": {
                "id": str(student["_id"]),
                "fullName": student["fullName"],
                "studentID": student["studentID"],
                "department": student.get("department"),
                "batch": student.get("batch"),
                "class_year": student.get("class_year"),
                "semester": student.get("semester"),
                "registrationDate": student.get("registrationDate")
            },
            "attendance_history": sorted(attendance_history, key=lambda x: x["session_date"], reverse=True),
            "total_records": len(attendance_history)
        }

    except Exception as e:
        logger.error(f"Student details error: {e}")
        raise HTTPException(500, str(e))

# ====================== LIVE ATTENDANCE (WebSocket + cv2.VideoCapture) ======================

# WebSocket endpoint for live attendance frames streamed from the browser. It returns recognition results in real time.
@app.websocket("/attendance/live")
async def live_attendance(websocket: WebSocket):
    """
    Live attendance via client-sent frames.
    - The browser captures frames via react-webcam and sends them as base64 JPEG
      in a JSON message: { "image": "data:image/jpeg;base64,..." }
    - This endpoint decodes each frame, runs InsightFace + emotion + pose,
      and streams JSON recognition results back to the client.
    - No server-side camera is required.
    """
    await websocket.accept()
    logger.info("🎥 Live attendance WebSocket connected")

    # Send a ready signal so the frontend knows it can start sending frames
    await websocket.send_json({"status": "ready", "message": "Send frames as base64 JPEG"})

    try:
        while True:
            # ── Wait for a frame from the browser ────────
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # No frame received in 30 s – send a heartbeat and keep waiting
                await websocket.send_json({"status": "waiting"})
                continue

            # ── Decode the incoming JSON ──────────────────
            try:
                msg = json.loads(raw)
            except Exception:
                await websocket.send_json({"status": "error", "message": "Invalid JSON"})
                continue

            image_b64 = msg.get("image", "")
            if not image_b64:
                await websocket.send_json({"status": "error", "message": "No image field in message"})
                continue

            # ── Decode base64 → PIL Image ─────────────────
            try:
                if "," in image_b64:
                    image_b64 = image_b64.split(",", 1)[1]
                img_bytes = base64.b64decode(image_b64)
                pil_image = Image.open(BytesIO(img_bytes)).convert("RGB")
            except Exception as decode_err:
                logger.warning(f"Frame decode error: {decode_err}")
                await websocket.send_json({"status": "error", "message": "Invalid image data"})
                continue

            # ── Face detection ────────────────────────────
            try:
                faces = detect_faces_for_attendance(pil_image)
            except Exception as det_err:
                logger.error(f"Detection error: {det_err}")
                faces = []

            results = []
            image_width = pil_image.width
            image_height = pil_image.height

            for face in faces:
                try:
                    x1, y1, x2, y2 = face["bbox"]
                    face_crop = pil_image.crop((x1, y1, x2, y2))

                    # Recognition + Emotion
                    recog_emotion = predict_recog_emotion(
                        face_crop, embedding=face.get("embedding")
                    )

                    # Pose
                    pose_result = predict_pose(face, image_width, image_height)

                    results.append({
                        "bbox": face["bbox"],
                        "student_id": recog_emotion["student_id"],
                        "full_name": recog_emotion["full_name"],
                        "emotion": recog_emotion["emotion"],
                        "pose": pose_result["pose"],
                        "recognized": recog_emotion["recognized"],
                        "pose_confidence": pose_result.get("pose_confidence", 60.0)
                    })
                except Exception as face_err:
                    logger.warning(f"Face processing error: {face_err}")
                    continue

            # ── Send results ──────────────────────────────
            present = [r["full_name"] for r in results if r["recognized"]]
            payload = {
                "status": "success",
                "total_faces_detected": len(faces),
                "recognized_count": len(present),
                "results": results,
                "present_list": present
            }

            try:
                await websocket.send_json(payload)
            except Exception:
                # Client disconnected during send
                break

    except WebSocketDisconnect:
        logger.info("🔌 Live attendance client disconnected")
    except Exception as e:
        logger.error(f"Live attendance error: {e}", exc_info=True)
        try:
            await websocket.send_json({"status": "error", "message": str(e)})
        except Exception:
            pass


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
