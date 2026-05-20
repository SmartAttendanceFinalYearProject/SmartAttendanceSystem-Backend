import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import insightface
import cv2
import pickle

# ========================= CONFIG =========================
DATA_PATH = Path("dataset/raw_data/student_photos")
MODEL_SAVE_PATH = "models/student_recognizer.pth"
EMBEDDINGS_PATH = "models/embeddings.pkl"

BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load InsightFace for embedding extraction
print("Loading InsightFace buffalo_l...")
face_app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# ========================= DATASET =========================
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# Extract embeddings once and save
def extract_all_embeddings():
    print("Extracting embeddings from all student photos...")
    embeddings = []
    labels = []
    student_mapping = {}
    label_id = 0

    for student_dir in sorted(DATA_PATH.iterdir()):
        if not student_dir.is_dir() or not student_dir.name.startswith("student_"):
            continue

        recog_dir = student_dir / "recognition"
        if not recog_dir.exists():
            continue

        student_mapping[student_dir.name] = label_id
        print(f"Processing {student_dir.name}...")

        for img_path in recog_dir.glob("*.jpg"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            faces = face_app.get(img)
            if len(faces) > 0:
                # Take the face with highest confidence
                best_face = max(faces, key=lambda x: x.det_score if x.det_score is not None else 0)
                emb = best_face.normed_embedding.astype(np.float32)
                embeddings.append(emb)
                labels.append(label_id)

        label_id += 1

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    # Save for future use
    os.makedirs("models", exist_ok=True)
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump({"embeddings": embeddings, "labels": labels, "mapping": student_mapping}, f)

    print(f"Extracted {len(embeddings)} embeddings from {label_id} students")
    return embeddings, labels, student_mapping

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

# ========================= MAIN =========================
if __name__ == "__main__":
    # Extract embeddings (only runs first time)
    if not os.path.exists(EMBEDDINGS_PATH):
        embeddings, labels, student_mapping = extract_all_embeddings()
    else:
        print("Loading pre-extracted embeddings...")
        with open(EMBEDDINGS_PATH, "rb") as f:
            data = pickle.load(f)
            embeddings, labels, student_mapping = data["embeddings"], data["labels"], data["mapping"]

    # Create dataset and dataloader
    dataset = EmbeddingDataset(embeddings, labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Model
    model = RecognizerHead(num_classes=len(student_mapping)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    print(f"Starting training on {len(student_mapping)} students...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for emb, lbl in tqdm(dataloader):
            emb, lbl = emb.to(device), lbl.to(device)

            optimizer.zero_grad()
            outputs = model(emb)
            loss = criterion(outputs, lbl)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += lbl.size(0)
            correct += predicted.eq(lbl).sum().item()

        accuracy = 100. * correct / total
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {running_loss/len(dataloader):.4f} | Accuracy: {accuracy:.2f}%")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': len(student_mapping),
        'student_mapping': student_mapping
    }, MODEL_SAVE_PATH)

    print(f"\n✅ Training Completed! Model saved to: {MODEL_SAVE_PATH}")