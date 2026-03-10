from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime 
from pydantic import BaseModel, Field, EmailStr  

class AttendanceStatus(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"

class EmotionStatus(str, Enum):
    HAPPY = "happy"
    NEUTRAL = "neutral"
    SAD = "sad"
    ANGRY = "angry"
    UNKNOWN = "unknown"

class PoseStatus(str, Enum):
    UP = "looking_up"
    DOWN = "looking_down"
    FRONT = "looking_front"
    UNKNOWN = "unknown"

class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"

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

class StudentCreate(BaseModel):
    fullName: str = Field(..., min_length=2, max_length=100)
    studentID: str = Field(..., min_length=3, max_length=50)
    department: Optional[str] = Field(None, max_length=100)
    section: Optional[str] = Field(None, max_length=50)
    email: Optional[EmailStr] = None
    # faceEmbedding is handled separately

class StudentOut(BaseModel):
    id: str
    fullName: str
    studentID: str
    department: Optional[str] = None
    section: Optional[str] = None
    email: Optional[str] = None
    registrationDate: datetime

    class Config:
        from_attributes = True