import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import insightface

# ========================= CONFIG =========================
DATA_PATH = Path("dataset/raw_data/student_photos")
MODEL_SAVE_PATH = "models/emotion_model.pth"
EMOTION_CLASSES = ["neutral", "happy", "angry"]

BATCH_SIZE = 32
NUM_EPOCHS = 40
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load InsightFace
print("Loading InsightFace...")
face_app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# ========================= DATASET =========================
class EmotionDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(EMOTION_CLASSES)}

        print("Loading emotion dataset...")

        for student_dir in sorted(DATA_PATH.iterdir()):
            if not student_dir.is_dir() or not student_dir.name.startswith("student_"):
                continue

            # Neutral images from recognition folder
            recog_dir = student_dir / "recognition"
            if recog_dir.exists():
                for img_path in recog_dir.glob("*.jpg"):
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx["neutral"])

            # Happy & Angry from emotion folder
            emotion_dir = student_dir / "emotion"
            if emotion_dir.exists():
                for img_path in emotion_dir.glob("*.jpg"):
                    filename = img_path.name.lower()
                    if "happy" in filename:
                        label = self.class_to_idx["happy"]
                    elif "angry" in filename:
                        label = self.class_to_idx["angry"]
                    else:
                        continue
                    self.image_paths.append(img_path)
                    self.labels.append(label)

        print(f"Loaded {len(self.image_paths)} images | Classes: {EMOTION_CLASSES}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            # Fallback: return black image
            image = np.zeros((112, 112, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Try to align face
        faces = face_app.get(image)
        if len(faces) > 0 and hasattr(faces[0], 'aligned') and faces[0].aligned is not None:
            aligned_face = faces[0].aligned
            pil_image = Image.fromarray(aligned_face)
        else:
            # Fallback if alignment fails
            pil_image = Image.fromarray(image)

        if self.transform:
            pil_image = self.transform(pil_image)

        return pil_image, label


# Data Transforms
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ========================= MODEL =========================
class EmotionModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# ========================= TRAINING =========================
if __name__ == "__main__":
    dataset = EmotionDataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    model = EmotionModel(num_classes=len(EMOTION_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    print("Starting Emotion Model Training...\n")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': EMOTION_CLASSES
    }, MODEL_SAVE_PATH)

    print(f"\n✅ Emotion Training Completed! Model saved to: {MODEL_SAVE_PATH}")