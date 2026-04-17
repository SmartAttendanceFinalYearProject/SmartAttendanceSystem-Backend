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
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# Load the model
def load_pose_model():
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    classes = checkpoint.get('classes', ['standing', 'sitting'])
    
    model = PoseModel(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Pose model loaded | Classes: {classes}")
    return model, classes


pose_model, POSE_CLASSES = load_pose_model()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict_pose(image: Image.Image) -> dict:
    """Predict sitting or standing using upper body crop"""
    try:
        # Convert to OpenCV
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Use larger upper body crop for better sitting/standing detection
        h, w = img_cv.shape[:2]
        body_crop = img_cv[0:int(h * 0.72), :, :]   # Upper 72% of image
        
        pil_image = Image.fromarray(cv2.cvtColor(body_crop, cv2.COLOR_BGR2RGB))

        input_tensor = transform(pil_image).unsqueeze(0).to(device)

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
        return {"pose": "standing", "pose_confidence": 0.0}