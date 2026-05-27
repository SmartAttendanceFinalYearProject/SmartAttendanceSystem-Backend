import torch
import torch.nn as nn
from torchvision import transforms, models  
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import insightface
import pickle

# ========================= CONFIG =========================
MODEL_PATH = Path("models/multitask_model.pth")
EMBEDDINGS_PATH = Path("models/embeddings.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load InsightFace
face_app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Load student mapping
with open(EMBEDDINGS_PATH, "rb") as f:
    data = pickle.load(f)
    student_mapping = data['mapping']
    id_to_student = {v: k for k, v in student_mapping.items()}

# ========================= MULTI-TASK MODEL =========================
class MultiTaskModel(nn.Module):
    def __init__(self, num_students):
        super().__init__()
        self.backbone = models.resnet18(weights=None)   # Fixed
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.student_head = nn.Linear(self.feature_dim, num_students)
        self.emotion_head = nn.Linear(self.feature_dim, 3)   # neutral, happy, angry
        self.pose_head = nn.Linear(self.feature_dim, 2)      # standing, sitting

    def forward(self, x):
        features = self.backbone(x)
        return (
            self.student_head(features),
            self.emotion_head(features),
            self.pose_head(features)
        )


# Load the trained model
checkpoint = torch.load(MODEL_PATH, map_location=device)
model = MultiTaskModel(num_students=checkpoint['num_students']).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("✅ Multi-Task Model Loaded Successfully!")

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

EMOTION_CLASSES = ["neutral", "happy", "angry"]
POSE_CLASSES = ["standing", "sitting"]


def predict_all(image: Image.Image):
    """One model → Student + Emotion + Pose"""
    try:
        # Convert to cv2
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Re-detect face on the cropped image for better alignment
        faces = face_app.get(img_cv)
        
        if not faces:
            # Fallback: use original crop
            input_tensor = transform(image).unsqueeze(0).to(device)
        else:
            best_face = max(faces, key=lambda f: f.det_score or 0)
            aligned = best_face.aligned if hasattr(best_face, 'aligned') and best_face.aligned is not None else image
            input_tensor = transform(aligned).unsqueeze(0).to(device)

        with torch.no_grad():
            student_out, emotion_out, pose_out = model(input_tensor)

        # Student Recognition
        student_idx = student_out.argmax(1).item()
        student_name = id_to_student.get(student_idx, "Unknown")

        # Emotion
        emotion_idx = emotion_out.argmax(1).item()
        emotion = EMOTION_CLASSES[emotion_idx]

        # Pose
        pose_idx = pose_out.argmax(1).item()
        pose = POSE_CLASSES[pose_idx]

        return {
            "student_id": student_name,
            "full_name": student_name,
            "emotion": emotion,
            "pose": pose,
            "recognized": student_name != "Unknown"
        }

    except Exception as e:
        print(f"Multi-task prediction error: {e}")
        return None
    """One model → Student + Emotion + Pose"""
    try:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        faces = face_app.get(img_cv)

        if not faces:
            return None

        best_face = max(faces, key=lambda f: f.det_score or 0)
        aligned_face = best_face.aligned if hasattr(best_face, 'aligned') and best_face.aligned is not None else image

        input_tensor = transform(aligned_face).unsqueeze(0).to(device)

        with torch.no_grad():
            student_out, emotion_out, pose_out = model(input_tensor)

        student_idx = student_out.argmax(1).item()
        student_name = id_to_student.get(student_idx, "Unknown")

        emotion_idx = emotion_out.argmax(1).item()
        emotion = EMOTION_CLASSES[emotion_idx]

        pose_idx = pose_out.argmax(1).item()
        pose = POSE_CLASSES[pose_idx]

        return {
            "student_id": student_name,
            "full_name": student_name,
            "emotion": emotion,
            "pose": pose,
            "recognized": student_name != "Unknown"
        }

    except Exception as e:
        print(f"Multi-task prediction error: {e}")
        return None