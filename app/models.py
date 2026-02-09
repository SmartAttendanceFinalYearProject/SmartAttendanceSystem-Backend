from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List

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
    OTHER = "other"

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

class UserCreate(BaseModel):
    name: str = Field(..., min_length=2)
    student_id: Optional[str] = None
    group: Optional[str] = None

class UserOut(BaseModel):
    id: str
    name: str
    student_id: Optional[str]
    group: Optional[str]