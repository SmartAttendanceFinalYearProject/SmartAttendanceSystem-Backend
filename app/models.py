from enum import Enum
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List
from datetime import datetime 
import uuid

# ====================== OLD DETECTION MODELS (Required by main.py) ======================
class FaceBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

class DetectionRequest(BaseModel):
    image_base64: str

class DetectionResponse(BaseModel):
    total_faces: int
    faces: List[FaceBox]
    message: str


# ====================== STUDENT ======================
class StudentCreate(BaseModel):
    fullName: str = Field(..., min_length=2, max_length=100)
    studentID: str = Field(..., min_length=3, max_length=50)
    department: Optional[str] = Field(None, max_length=100)
    section: Optional[str] = Field(None, max_length=50)
    email: Optional[EmailStr] = None
    batch: str = Field(..., description="e.g. 2022, 2023")
    class_year: str = Field(..., description="e.g. 1st, 2nd, 3rd, 4th")
    semester: str = Field(..., description="e.g. 1st, 2nd")


class StudentOut(BaseModel):
    id: str
    fullName: str
    studentID: str
    department: Optional[str]
    section: Optional[str]
    email: Optional[str]
    batch: str
    class_year: str
    semester: str
    registrationDate: datetime

class StudentMinimal(BaseModel):
    id: str
    fullName: str
    studentID: str


# ====================== SUBJECT ======================
class SubjectCreate(BaseModel):
    subject_name: str
    subject_code: str

class SubjectOut(BaseModel):
    id: str
    subject_name: str
    subject_code: str

class SubjectUpdate(BaseModel):
    subject_name: Optional[str] = None
    subject_code: Optional[str] = None


# ====================== TEACHER ======================
class TeacherCreate(BaseModel):
    full_name: str
    subject_id: str
    username: str
    password: str

class TeacherOut(BaseModel):
    id: str
    full_name: str
    subject_id: str
    username: str

class TeacherUpdate(BaseModel):
    full_name: Optional[str] = None
    subject_id: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None


# ====================== ATTENDANCE ======================
class AttendanceRecord(BaseModel):
    student_id: str
    full_name: str
    status: str  # "present" or "absent"
    emotion: Optional[str] = None
    pose: Optional[str] = None
    recognition_confidence: Optional[float] = None
    emotion_confidence: Optional[float] = None
    pose_confidence: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class AttendanceSession(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_date: datetime = Field(default_factory=datetime.utcnow)
    records: List[AttendanceRecord]


# ====================== CLASS ======================
class DaySchedule(BaseModel):
    day: str = Field(..., example="Monday")
    start_time: str = Field(..., example="10:00 AM")
    end_time: str = Field(..., example="11:30 AM")

class ClassSchedule(BaseModel):
    schedule: List[DaySchedule]

class ClassCreate(BaseModel):
    class_name: str
    subject_id: str
    teacher_id: str
    start_date: datetime
    end_date: datetime
    schedule: ClassSchedule
    students: List[str] = []
    attendance_sessions: List[AttendanceSession] = []

class ClassOut(BaseModel):
    id: str
    class_name: str
    subject_id: str
    subject_name: Optional[str] = None
    subject_code: Optional[str] = None
    teacher_id: str
    teacher_name: str
    start_date: datetime
    end_date: datetime
    schedule: ClassSchedule
    student_count: int = 0
    students: List[str] = []
    student_details: List[StudentMinimal] = []
    attendance_sessions: List[AttendanceSession] = []

class ClassUpdate(BaseModel):
    class_name: Optional[str] = None
    subject_id: Optional[str] = None
    teacher_id: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    schedule: Optional[ClassSchedule] = None
    students: Optional[List[str]] = None
    attendance_sessions: Optional[List[AttendanceSession]] = None


# ====================== AUTH ======================
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: Optional[str] = None
    full_name: Optional[str] = None

class TokenData(BaseModel):
    username: Optional[str] = None
    role: str = Field(..., pattern="^(admin|teacher)$")

class UserLogin(BaseModel):
    username: str
    password: str

class TeacherCreateByAdmin(BaseModel):
    full_name: str
    subject_id: str
    username: str
    password: str