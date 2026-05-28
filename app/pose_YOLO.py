import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
from pathlib import Path

# ========================= CONFIG =========================
MODEL_PATH = Path("models/best_pose_model.pth")   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

POSE_CLASSES = ["sitting", "standing"]

# ========================= MODEL =========================
class PoseModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.backbone(x)


# Load the trained model
checkpoint = torch.load(MODEL_PATH, map_location=device)
model = PoseModel(num_classes=2).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Trained Pose Model Loaded Successfully! Classes: {POSE_CLASSES}")

# ========================= TRANSFORM =========================
pose_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_pose(image: Image.Image) -> dict:
    """Use trained ResNet18 model for sitting vs standing"""
    try:
        # Transform the face crop
        input_tensor = pose_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_idx = outputs.argmax(1).item()
            confidence = float(probabilities[predicted_idx] * 100)

        pose_label = POSE_CLASSES[predicted_idx]

        return {
            "pose": pose_label,
            "pose_confidence": round(confidence, 1)
        }

    except Exception as e:
        print(f"Pose model error: {e}")
        return {"pose": "standing", "pose_confidence": 50.0}