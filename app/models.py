from enum import Enum
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List
from datetime import datetime 

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
    batch_class_year: str = Field(..., description="e.g. 2022, 2023, BCY2024")


class StudentOut(BaseModel):
    id: str
    fullName: str
    studentID: str
    department: Optional[str]
    section: Optional[str]
    email: Optional[str]
    batch_class_year: str
    registrationDate: datetime


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


# ====================== CLASS ======================
class ClassSchedule(BaseModel):
    days: List[str] = Field(..., example=["Monday", "Wednesday", "Friday"])
    start_time: str = Field(..., example="10:00 AM")
    end_time: str = Field(..., example="11:30 AM")


class ClassCreate(BaseModel):
    class_name: str
    subject_id: str
    teacher_name: str
    start_date: datetime
    end_date: datetime
    schedule: ClassSchedule

class ClassOut(BaseModel):
    id: str
    class_name: str
    subject_id: str
    teacher_name: str
    start_date: datetime
    end_date: datetime
    schedule: ClassSchedule
    student_count: int = 0


class ClassUpdate(BaseModel):
    class_name: Optional[str] = None
    subject_id: Optional[str] = None
    teacher_name: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    schedule: Optional[ClassSchedule] = None


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