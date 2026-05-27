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
MODEL_PATH = "models/train_recognition_emotion_finetune.pth"   
NEW_MODEL_PATH = "models/recog_emotion_model_finetuned.pth"

BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 0.0005   # Small learning rate for fine-tuning

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========================= TRANSFORMS =========================
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ========================= EMOTION DATASET (Only Emotion) =========================
class EmotionFineTuneDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.samples = []   # (img_path, emotion_label)

        emotion_train = DATA_PATH / "emotion" / "train"
        emotion_map = {"neutral": 0, "happy": 1, "angry": 2}

        print("Loading emotion dataset for fine-tuning...")

        for emotion_name, label in emotion_map.items():
            folder = emotion_train / emotion_name
            if not folder.exists():
                print(f"Warning: {folder} not found")
                continue

            for img_path in folder.rglob("*.*"):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                self.samples.append((img_path, label))

        print(f"✅ Loaded {len(self.samples)} emotion images "
              f"({len([s for s in self.samples if s[1]==2])} angry)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, emotion_label = self.samples[idx]
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                image = np.zeros((112, 112, 3), dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image)
            if self.transform:
                pil_img = self.transform(pil_img)
            return pil_img, emotion_label
        except:
            blank = torch.zeros((3, 112, 112))
            return blank, emotion_label


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


# ========================= FINE-TUNING =========================
if __name__ == "__main__":
    # Load previous model
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = RecogEmotionModel(checkpoint['num_students']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Loaded previous model for fine-tuning")

    # Freeze backbone + student head
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.student_head.parameters():
        param.requires_grad = False

    # Only train emotion head
    optimizer = optim.Adam(model.emotion_head.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    dataset = EmotionFineTuneDataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    print(f"\n🚀 Fine-tuning Emotion Head only ({NUM_EPOCHS} epochs)...\n")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, emotion_labels in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            images = images.to(device)
            emotion_labels = emotion_labels.to(device)

            optimizer.zero_grad()
            _, emotion_out = model(images)
            loss = criterion(emotion_out, emotion_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total += images.size(0)
            correct += (emotion_out.argmax(1) == emotion_labels).sum().item()

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {total_loss/len(dataloader):.4f} | "
              f"Emotion Acc: {100*correct/total:.2f}%")

    # Save fine-tuned model
    os.makedirs("models", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_students': checkpoint['num_students'],
    }, NEW_MODEL_PATH)

    print(f"\n✅ Fine-tuning Completed! Model saved to: {NEW_MODEL_PATH}")