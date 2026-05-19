import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import insightface

# Use the reliable database recognizer
from .recognizer import recognize_student

# ========================= CONFIG =========================
MODEL_PATH = Path("models/recog_emotion_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load InsightFace
face_app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# ========================= MULTI-TASK MODEL (Match your saved checkpoint) =========================
class MultiTaskModel(nn.Module):
    def __init__(self, num_students):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.student_head = nn.Linear(self.feature_dim, num_students)
        self.emotion_head = nn.Linear(self.feature_dim, 3)   # neutral, happy, angry

    def forward(self, x):
        features = self.backbone(x)
        return self.student_head(features), self.emotion_head(features)


# Load the trained model
checkpoint = torch.load(MODEL_PATH, map_location=device)
model = MultiTaskModel(num_students=checkpoint['num_students']).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("✅ Recognition + Emotion Model Loaded Successfully!")

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

EMOTION_CLASSES = ["neutral", "happy", "angry"]


def predict_recog_emotion(image: Image.Image):
    """Recognition (Database) + Emotion (Model)"""
    try:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        faces = face_app.get(img_cv)

        if not faces:
            return {
                "student_id": "Unknown",
                "full_name": "Not registered as student",
                "emotion": "neutral",
                "recognized": False
            }

        best_face = max(faces, key=lambda f: f.det_score or 0)
        aligned = best_face.aligned if hasattr(best_face, 'aligned') and best_face.aligned is not None else image

        # === Emotion ===
        input_tensor = transform(aligned).unsqueeze(0).to(device)
        with torch.no_grad():
            student_out, emotion_out = model(input_tensor)

        emotion_idx = emotion_out.argmax(1).item()
        emotion = EMOTION_CLASSES[emotion_idx]

        # === Recognition (Reliable DB method) ===
        embedding = best_face.normed_embedding
        recog = recognize_student(embedding)

        return {
            "student_id": recog["student_id"],
            "full_name": recog["full_name"],
            "emotion": emotion,
            "recognized": recog["recognized"]
        }

    except Exception as e:
        print(f"Recog-Emotion error: {e}")
        return {
            "student_id": "Unknown",
            "full_name": "Not registered as student",
            "emotion": "neutral",
            "recognized": False
        }