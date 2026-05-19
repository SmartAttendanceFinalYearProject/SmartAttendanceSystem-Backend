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

# ========================= CONFIG =========================
DATA_PATH = Path("dataset/raw_data/student_photos")

MODEL_SAVE_PATH = "models/recog_emotion_model_1.pth"

BATCH_SIZE = 8
NUM_EPOCHS = 15
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========================= TRANSFORMS =========================
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ========================= DATASET =========================
class RecogEmotionDataset(Dataset):
    def __init__(self, transform=None, is_emotion=False):
        self.transform = transform
        self.is_emotion = is_emotion
        self.samples = []

        if not is_emotion:
            # Recognition Data - for student head
            recog_train = DATA_PATH / "recognition" / "train"
            student_to_id = {}
            sid = 0

            if recog_train.exists():
                for student_folder in sorted(recog_train.iterdir()):
                    if not student_folder.is_dir(): continue
                    student_name = student_folder.name
                    if student_name not in student_to_id:
                        student_to_id[student_name] = sid
                        sid += 1
                    s_id = student_to_id[student_name]

                    for img_path in student_folder.rglob("*.*"):
                        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}: continue
                        self.samples.append((img_path, s_id, 0))   # dummy emotion

            self.num_students = len(student_to_id)
        else:
            # Emotion Data - for emotion head only
            emotion_train = DATA_PATH / "emotion" / "train"
            emotion_map = {"neutral": 0, "happy": 1, "angry": 2}

            if emotion_train.exists():
                for emotion_name, label in emotion_map.items():
                    folder = emotion_train / emotion_name
                    if not folder.exists(): continue
                    for img_path in folder.rglob("*.*"):
                        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}: continue
                        self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label1, label2 = self.samples[idx] if not self.is_emotion else (self.samples[idx][0], 0, self.samples[idx][1])
        
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                image = np.zeros((112, 112, 3), dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image)
            if self.transform:
                pil_img = self.transform(pil_img)
            return pil_img, label1, label2
        except:
            blank = torch.zeros((3, 112, 112))
            return blank, label1, label2


# ========================= MODEL =========================
class RecogEmotionModel(nn.Module):
    def __init__(self, num_students):
        super().__init__()
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.student_head = nn.Linear(feature_dim, num_students)
        self.emotion_head = nn.Linear(feature_dim, 3)

    def forward(self, x):
        features = self.backbone(x)
        return self.student_head(features), self.emotion_head(features)


# ========================= TRAINING =========================
if __name__ == "__main__":
    # Two separate datasets
    recog_dataset = RecogEmotionDataset(transform=transform, is_emotion=False)
    emotion_dataset = RecogEmotionDataset(transform=transform, is_emotion=True)

    recog_loader = DataLoader(recog_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    emotion_loader = DataLoader(emotion_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)

    model = RecogEmotionModel(recog_dataset.num_students).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🚀 Starting Training...\nRecognition samples: {len(recog_dataset)} | Emotion samples: {len(emotion_dataset)}\n")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        # === Train Student Head ===
        for images, student_labels, _ in tqdm(recog_loader, desc=f"Epoch {epoch+1} [Student]"):
            images = images.to(device)
            student_labels = student_labels.to(device)

            optimizer.zero_grad()
            student_out, _ = model(images)
            loss = criterion(student_out, student_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # === Train Emotion Head ===
        for images, _, emotion_labels in tqdm(emotion_loader, desc=f"Epoch {epoch+1} [Emotion]"):
            images = images.to(device)
            emotion_labels = emotion_labels.to(device)

            optimizer.zero_grad()
            _, emotion_out = model(images)
            loss = criterion(emotion_out, emotion_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Completed | Avg Loss: {total_loss/(len(recog_loader)+len(emotion_loader)):.4f}")

    # Save Model
    os.makedirs("models", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_students': recog_dataset.num_students,
    }, MODEL_SAVE_PATH)

    print(f"\n✅ Training Completed! Model saved to: {MODEL_SAVE_PATH}")