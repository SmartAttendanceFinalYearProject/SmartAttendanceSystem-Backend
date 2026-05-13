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
MODEL_SAVE_PATH = "models/recog_emotion_model.pth"
CHECKPOINT_PATH = "models/recog_emotion_checkpoint.pth"

BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# InsightFace for face alignment
face_app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Strong data augmentation
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ========================= DATASET =========================
class RecogEmotionDataset(Dataset):
    def __init__(self, transform=None, split="train"):
        self.transform = transform
        self.samples = []   # (image_path, student_id, emotion_label)

        student_to_id = {}
        sid = 0

        base_path = DATA_PATH / split

        for student_dir in sorted(base_path.iterdir()):
            if not student_dir.is_dir():
                continue

            if student_dir.name not in student_to_id:
                student_to_id[student_dir.name] = sid
                sid += 1
            s_id = student_to_id[student_dir.name]

            # === Recognition Folder (Neutral) ===
            recog_dir = student_dir / "recognition"
            if recog_dir.exists():
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG", "*.heic"]:
                    for img_path in recog_dir.glob(ext):
                        self.samples.append((img_path, s_id, 0))  # 0 = neutral

            # === Emotion Folder ===
            emotion_dir = student_dir / "emotion"
            if emotion_dir.exists():
                emotion_map = {"neutral": 0, "happy": 1, "angry": 2}
                for emotion_name, label in emotion_map.items():
                    cls_dir = emotion_dir / emotion_name
                    if cls_dir.exists():
                        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG", "*.heic"]:
                            for img_path in cls_dir.glob(ext):
                                self.samples.append((img_path, s_id, label))

        print(f"[{split.upper()}] Loaded {len(self.samples)} images | Students: {len(student_to_id)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, student_label, emotion_label = self.samples[idx]

        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Face alignment using buffalo_l
        faces = face_app.get(image)
        if len(faces) > 0 and hasattr(faces[0], 'aligned') and faces[0].aligned is not None:
            pil_img = Image.fromarray(faces[0].aligned)
        else:
            pil_img = Image.fromarray(image)

        if self.transform:
            pil_img = self.transform(pil_img)

        return pil_img, student_label, emotion_label


# ========================= MODEL =========================
class RecogEmotionModel(nn.Module):
    def __init__(self, num_students):
        super().__init__()
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.student_head = nn.Linear(self.feature_dim, num_students)
        self.emotion_head = nn.Linear(self.feature_dim, 3)

    def forward(self, x):
        features = self.backbone(x)
        return self.student_head(features), self.emotion_head(features)


# ========================= TRAINING =========================
if __name__ == "__main__":
    train_dataset = RecogEmotionDataset(transform=transform, split="train")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)

    num_students = len(set(s[1] for s in train_dataset.samples))

    model = RecogEmotionModel(num_students).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    patience_counter = 0

    print(f"Starting Training | Students: {num_students} | Epochs: {NUM_EPOCHS} | Batch Size: {BATCH_SIZE}")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        correct_student = correct_emotion = 0
        total = 0

        for images, student_labels, emotion_labels in tqdm(train_loader):
            images = images.to(device)
            student_labels = student_labels.to(device)
            emotion_labels = emotion_labels.to(device)

            optimizer.zero_grad()
            student_out, emotion_out = model(images)

            loss = criterion(student_out, student_labels) + criterion(emotion_out, emotion_labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total += images.size(0)

            correct_student += (student_out.argmax(1) == student_labels).sum().item()
            correct_emotion += (emotion_out.argmax(1) == emotion_labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        student_acc = 100 * correct_student / total
        emotion_acc = 100 * correct_emotion / total

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {avg_loss:.4f} | "
              f"Student Acc: {student_acc:.2f}% | Emotion Acc: {emotion_acc:.2f}%")

        # Early Stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_students': num_students,
            }, MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("✅ Early stopping triggered!")
                break

    print(f"\n🎉 Training Completed! Model saved to: {MODEL_SAVE_PATH}")