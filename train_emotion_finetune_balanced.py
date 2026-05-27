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
from collections import Counter

# ========================= CONFIG =========================
DATA_PATH = Path("dataset/raw_data/student_photos")
MODEL_PATH = "models/recog_emotion_model_finetuned.pth"   # Use your last fine-tuned model
NEW_MODEL_PATH = "models/recog_emotion_model_final.pth"

BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 0.0003

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========================= TRANSFORMS =========================
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ========================= DATASET =========================
class EmotionDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.samples = []
        emotion_train = DATA_PATH / "emotion" / "train"
        emotion_map = {"neutral": 0, "happy": 1, "angry": 2}

        for emotion_name, label in emotion_map.items():
            folder = emotion_train / emotion_name
            if not folder.exists(): continue
            for img_path in folder.rglob("*.*"):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}: continue
                self.samples.append((img_path, label))

        # Count samples per class
        labels = [s[1] for s in self.samples]
        count = Counter(labels)
        print(f"✅ Loaded: Neutral={count[0]}, Happy={count[1]}, Angry={count[2]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                image = np.zeros((112, 112, 3), dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image)
            if self.transform:
                pil_img = self.transform(pil_img)
            return pil_img, label
        except:
            return torch.zeros((3, 112, 112)), label


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
    dataset = EmotionDataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Load previous model
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = RecogEmotionModel(checkpoint['num_students']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Loaded model for balanced fine-tuning")

    # Freeze everything except emotion head
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.student_head.parameters():
        param.requires_grad = False

    # === Class Weights (Very Important) ===
    # Give higher weight to neutral because angry is over-predicted
    class_weights = torch.tensor([1.5, 1.0, 0.8], dtype=torch.float32).to(device)  # neutral, happy, angry
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.emotion_head.parameters(), lr=LEARNING_RATE)

    print("\n🚀 Starting Balanced Fine-tuning...\n")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            _, emotion_out = model(images)
            loss = criterion(emotion_out, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total += images.size(0)
            correct += (emotion_out.argmax(1) == labels).sum().item()

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {total_loss/len(dataloader):.4f} | "
              f"Accuracy: {100*correct/total:.2f}%")

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_students': checkpoint['num_students'],
    }, NEW_MODEL_PATH)

    print(f"\n✅ Balanced Fine-tuning Completed! Saved as: {NEW_MODEL_PATH}")
    