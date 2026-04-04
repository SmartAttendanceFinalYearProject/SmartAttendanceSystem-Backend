import torch
import torch.nn as nn
import pickle
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional

# ========================= CONFIG =========================
MODEL_PATH = Path("models/student_recognizer.pth")
EMBEDDINGS_PATH = Path("models/embeddings.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================= MODEL =========================
class RecognizerHead(nn.Module):
    def __init__(self, input_dim=512, num_classes=31):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# ========================= LOAD MODEL & MAPPING =========================
def load_recognizer():
    """Load the trained model and student mapping once"""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    # Load model
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = RecognizerHead(num_classes=checkpoint['num_classes']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load student mapping
    with open(EMBEDDINGS_PATH, "rb") as f:
        data = pickle.load(f)
        student_mapping = data['mapping']                    # e.g. {"student_001": 0, ...}
        id_to_student = {v: k for k, v in student_mapping.items()}  # reverse mapping

    print(f"✅ Recognizer loaded | {len(student_mapping)} students")
    return model, id_to_student


# Global variables (loaded once when module is imported)
recognizer_model, id_to_student = load_recognizer()


def recognize_student(embedding: np.ndarray) -> Dict[str, any]:
    """
    Takes one 512-dim embedding → returns student name + confidence
    """
    emb_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = recognizer_model(emb_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        confidence, predicted_id = torch.max(probs, dim=1)
    
    student_id = int(predicted_id.item())
    confidence_score = float(confidence.item())
    
    student_name = id_to_student.get(student_id, "Unknown")
    
    return {
        "student_id": student_name,
        "confidence": round(confidence_score * 100, 2),
        "recognized": confidence_score > 0.65   # you can adjust this threshold
    }


def recognize_faces_in_classroom(faces_list: List[dict]) -> List[dict]:
    """
    Takes the output of detect_faces_for_attendance() and adds student recognition
    """
    results = []
    for face in faces_list:
        if face.get("embedding") is None:
            continue
            
        emb = np.array(face["embedding"], dtype=np.float32)
        recog = recognize_student(emb)
        
        results.append({
            "bbox": face["bbox"],
            "confidence_detection": face["confidence"],
            "student_id": recog["student_id"],
            "recognition_confidence": recog["confidence"],
            "recognized": recog["recognized"],
            "landmarks": face.get("landmarks"),
            "pose": face.get("pose")
        })
    
    return results