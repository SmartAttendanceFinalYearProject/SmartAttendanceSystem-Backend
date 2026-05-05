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
from collections import defaultdict

# ========================= CONFIG =========================
DATA_PATH = Path("dataset/raw_data/student_photos")
MODEL_SAVE_PATH = "models/multitask_model.pth"
CHECKPOINT_PATH = "models/multitask_checkpoint.pth"

BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load InsightFace
face_app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# ========================= DATASET =========================
class MultiTaskDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.samples = []  # (image_path, student_label, emotion_label, pose_label)

        student_to_id = {}
        student_id = 0

        for student_dir in sorted(DATA_PATH.iterdir()):
            if not student_dir.is_dir() or not student_dir.name.startswith("student_"):
                continue

            if student_dir.name not in student_to_id:
                student_to_id[student_dir.name] = student_id
                student_id += 1

            s_id = student_to_id[student_dir.name]

            # Recognition images → Neutral
            recog_dir = student_dir / "recognition"
            if recog_dir.exists():
                for img_path in recog_dir.glob("*.jpg"):
                    self.samples.append((img_path, s_id, 0, 0))  # 0 = neutral, 0 = standing

            # Emotion folder
            emotion_dir = student_dir / "emotion"
            if emotion_dir.exists():
                for img_path in emotion_dir.glob("*.jpg"):
                    fname = img_path.name.lower()
                    emotion_label = 1 if "happy" in fname else 2 if "angry" in fname else 0
                    self.samples.append((img_path, s_id, emotion_label, 1))  # assume sitting

            # Pose folder
            pose_dir = student_dir / "pose"
            if pose_dir.exists():
                for img_path in pose_dir.glob("*.jpg"):
                    fname = img_path.name.lower()
                    pose_label = 1 if "sitting" in fname else 0  # 0=standing, 1=sitting
                    self.samples.append((img_path, s_id, 0, pose_label))

        print(f"Loaded {len(self.samples)} samples | Students: {len(student_to_id)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, student_label, emotion_label, pose_label = self.samples[idx]

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        faces = face_app.get(image)
        if len(faces) > 0 and faces[0].aligned is not None:
            pil_image = Image.fromarray(faces[0].aligned)
        else:
            pil_image = Image.fromarray(image)

        if self.transform:
            pil_image = self.transform(pil_image)

        return pil_image, student_label, emotion_label, pose_label


transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ========================= MULTI-TASK MODEL =========================
class MultiTaskModel(nn.Module):
    def __init__(self, num_students):
        super().__init__()
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()   # Remove final layer

        # Three heads
        self.student_head = nn.Linear(self.feature_dim, num_students)
        self.emotion_head = nn.Linear(self.feature_dim, 3)   # neutral, happy, angry
        self.pose_head = nn.Linear(self.feature_dim, 2)      # standing, sitting

    def forward(self, x):
        features = self.backbone(x)
        student_out = self.student_head(features)
        emotion_out = self.emotion_head(features)
        pose_out = self.pose_head(features)
        return student_out, emotion_out, pose_out


# ========================= TRAINING =========================
if __name__ == "__main__":
    dataset = MultiTaskDataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)

    num_students = max([s[1] for s in dataset.samples]) + 1
    model = MultiTaskModel(num_students=num_students).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting Multi-Task Training | Students: {num_students} | Samples: {len(dataset)}")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        correct_student = correct_emotion = correct_pose = 0
        total = 0

        for images, student_labels, emotion_labels, pose_labels in tqdm(dataloader):
            images = images.to(device)
            student_labels = student_labels.to(device)
            emotion_labels = emotion_labels.to(device)
            pose_labels = pose_labels.to(device)

            optimizer.zero_grad()
            student_out, emotion_out, pose_out = model(images)

            loss = (criterion(student_out, student_labels) + 
                    criterion(emotion_out, emotion_labels) + 
                    criterion(pose_out, pose_labels))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total += images.size(0)

            correct_student += (student_out.argmax(1) == student_labels).sum().item()
            correct_emotion += (emotion_out.argmax(1) == emotion_labels).sum().item()
            correct_pose += (pose_out.argmax(1) == pose_labels).sum().item()

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {total_loss/len(dataloader):.4f} | "
              f"Student Acc: {100*correct_student/total:.2f}% | "
              f"Emotion Acc: {100*correct_emotion/total:.2f}% | "
              f"Pose Acc: {100*correct_pose/total:.2f}%")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_students': num_students,
        'classes': {'emotion': ['neutral', 'happy', 'angry'], 'pose': ['standing', 'sitting']}
    }, MODEL_SAVE_PATH)

    print(f"\n✅ Multi-Task Model Training Completed! Saved to: {MODEL_SAVE_PATH}")