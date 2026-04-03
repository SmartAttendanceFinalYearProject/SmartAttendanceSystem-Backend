import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# ========================= CONFIG =========================
DATA_PATH = Path("dataset/raw_data/student_photos")
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "models/arcface_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# =======================================================

class StudentDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.student_to_label = {}
        label = 0

        for student_dir in sorted(self.root_dir.iterdir()):
            if student_dir.is_dir() and student_dir.name.startswith("student_"):
                recog_dir = student_dir / "recognition"
                if recog_dir.exists():
                    for img_path in recog_dir.glob("*.jpg"):
                        self.image_paths.append(img_path)
                        self.labels.append(label)
                    self.student_to_label[student_dir.name] = label
                    label += 1

        print(f"Loaded {len(self.image_paths)} images from {label} students")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, label


# Data transforms
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset
dataset = StudentDataset(DATA_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# Model: ResNet50 backbone + ArcFace head (simplified)
class ArcFaceModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 512)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.nn.functional.normalize(x, p=2, dim=1)  # L2 normalize
        x = self.fc(x)
        return x

model = ArcFaceModel(num_classes=len(dataset.student_to_label)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
print("Starting Training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader):
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

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {running_loss/len(dataloader):.4f} | Accuracy: {100.*correct/total:.2f}%")

# Save model
os.makedirs("models", exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'num_classes': len(dataset.student_to_label),
    'student_mapping': dataset.student_to_label
}, MODEL_SAVE_PATH)

print(f"\nTraining completed! Model saved to: {MODEL_SAVE_PATH}")