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
MODEL_SAVE_PATH = "models/pose_model.pth"
CHECKPOINT_PATH = "models/pose_checkpoint.pth"

BATCH_SIZE = 16
NUM_EPOCHS = 25
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load InsightFace for face alignment
print("Loading InsightFace...")
face_app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# ========================= AUTO DETECT POSE CLASSES =========================
def get_pose_classes():
    pose_classes = set()
    for student_dir in DATA_PATH.iterdir():
        if not student_dir.is_dir() or not student_dir.name.startswith("student_"):
            continue
        pose_dir = student_dir / "pose"
        if pose_dir.exists():
            for img_path in pose_dir.glob("*.jpg"):
                # Extract pose from filename (e.g., student001_front_looking_up_01.jpg → looking_up)
                name = img_path.name.lower()
                if "looking_up" in name:
                    pose_classes.add("looking_up")
                elif "looking_down" in name:
                    pose_classes.add("looking_down")
                elif "sitting" in name:
                    pose_classes.add("sitting")
                elif "standing" in name:
                    pose_classes.add("standing")
                else:
                    pose_classes.add("front_neutral")
    return sorted(list(pose_classes))

EMOTION_CLASSES = get_pose_classes()   # Wait, POSE_CLASSES
print(f"Detected Pose Classes: {EMOTION_CLASSES}")

# ========================= DATASET =========================
class PoseDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(EMOTION_CLASSES)}

        print("Loading pose dataset...")

        for student_dir in sorted(DATA_PATH.iterdir()):
            if not student_dir.is_dir() or not student_dir.name.startswith("student_"):
                continue

            pose_dir = student_dir / "pose"
            if pose_dir.exists():
                for img_path in pose_dir.glob("*.jpg"):
                    filename = img_path.name.lower()
                    if "looking_up" in filename:
                        label = self.class_to_idx["looking_up"]
                    elif "looking_down" in filename:
                        label = self.class_to_idx["looking_down"]
                    elif "sitting" in filename:
                        label = self.class_to_idx["sitting"]
                    elif "standing" in filename:
                        label = self.class_to_idx["standing"]
                    else:
                        label = self.class_to_idx["front_neutral"]
                    
                    self.image_paths.append(img_path)
                    self.labels.append(label)

        print(f"Loaded {len(self.image_paths)} pose images | Classes: {EMOTION_CLASSES}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(str(img_path))
        if image is None:
            image = np.zeros((112, 112, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Face alignment
        faces = face_app.get(image)
        if len(faces) > 0 and hasattr(faces[0], 'aligned') and faces[0].aligned is not None:
            pil_image = Image.fromarray(faces[0].aligned)
        else:
            pil_image = Image.fromarray(image)

        if self.transform:
            pil_image = self.transform(pil_image)

        return pil_image, label


# Data Transforms
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ========================= MODEL =========================
class PoseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# ========================= TRAINING =========================
if __name__ == "__main__":
    dataset = PoseDataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)

    model = PoseModel(num_classes=len(EMOTION_CLASSES)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    start_epoch = 0

    # Resume from checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        print(f"✅ Resuming from checkpoint...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resuming from epoch {start_epoch}")

    criterion = nn.CrossEntropyLoss()

    print(f"Starting Pose Training from epoch {start_epoch + 1}...\n")

    for epoch in range(start_epoch, NUM_EPOCHS):
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

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, CHECKPOINT_PATH)

    # Save final model
    os.makedirs("models", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': EMOTION_CLASSES
    }, MODEL_SAVE_PATH)

    print(f"\n✅ Pose Training Completed! Model saved to: {MODEL_SAVE_PATH}")