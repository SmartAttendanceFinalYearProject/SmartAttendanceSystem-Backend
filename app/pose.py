import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import insightface

# ========================= CONFIG =========================
MODEL_PATH = Path("models/pose_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load InsightFace
face_app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# ========================= MODEL =========================
class PoseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# Load model
def load_pose_model():
    # First detect classes from saved checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    classes = checkpoint.get('classes', ['front_neutral', 'looking_up', 'looking_down', 'sitting', 'standing'])
    
    model = PoseModel(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Pose model loaded | Classes: {classes}")
    return model, classes


pose_model, POSE_CLASSES = load_pose_model()

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def predict_pose(image: Image.Image) -> dict:
    """Predict pose from a face image"""
    try:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        faces = face_app.get(img_cv)

        if len(faces) > 0 and hasattr(faces[0], 'aligned') and faces[0].aligned is not None:
            face_img = Image.fromarray(faces[0].aligned)
        else:
            face_img = image

        input_tensor = transform(face_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = pose_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)

        pose = POSE_CLASSES[predicted_idx.item()]
        confidence_score = float(confidence.item()) * 100

        return {
            "pose": pose,
            "pose_confidence": round(confidence_score, 2)
        }

    except Exception as e:
        print(f"Pose prediction error: {e}")
        return {"pose": "front_neutral", "pose_confidence": 0.0}