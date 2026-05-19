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

MODEL_SAVE_PATH = "models/recog_emotion_model.pth"
CHECKPOINT_PATH = "models/recog_emotion_checkpoint.pth"

BATCH_SIZE = 4          
NUM_EPOCHS = 20          
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========================= TRANSFORMS =========================
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# ========================= DATASET =========================
class RecogEmotionDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.samples = []

        student_to_id = {}
        student_id = 0

        print("Scanning dataset folders...")

        # =====================================================
        # RECOGNITION DATASET
        # recognition/train/student001/img1.jpg
        # =====================================================
        recog_train = DATA_PATH / "recognition" / "train"

        if recog_train.exists():

            for student_folder in recog_train.iterdir():

                if not student_folder.is_dir():
                    continue

                student_name = student_folder.name

                if student_name not in student_to_id:
                    student_to_id[student_name] = student_id
                    student_id += 1

                s_id = student_to_id[student_name]

                for img_path in student_folder.rglob("*.*"):

                    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                        continue

                    # emotion label = neutral
                    self.samples.append((img_path, s_id, 0))

        # =====================================================
        # EMOTION DATASET
        # emotion/train/happy/img1.jpg
        # =====================================================
        emotion_train = DATA_PATH / "emotion" / "train"

        emotion_map = {
            "neutral": 0,
            "happy": 1,
            "angry": 2
        }

        if emotion_train.exists():

            for emotion_name, emotion_label in emotion_map.items():

                emotion_folder = emotion_train / emotion_name

                if not emotion_folder.exists():
                    continue

                for img_path in emotion_folder.rglob("*.*"):

                    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                        continue

                    # dummy student label
                    self.samples.append((img_path, 0, emotion_label))

        self.num_students = len(student_to_id)

        print(f"✅ Loaded {len(self.samples)} images")
        print(f"✅ Students found: {self.num_students}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_path, student_label, emotion_label = self.samples[idx]

        try:
            image = cv2.imread(str(img_path))

            if image is None:
                image = np.zeros((112, 112, 3), dtype=np.uint8)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            pil_image = Image.fromarray(image)

            if self.transform:
                pil_image = self.transform(pil_image)

            return pil_image, student_label, emotion_label

        except Exception as e:

            print(f"Error loading {img_path}: {e}")

            blank = torch.zeros((3, 112, 112), dtype=torch.float32)

            return blank, student_label, emotion_label


# ========================= MODEL =========================
class RecogEmotionModel(nn.Module):

    def __init__(self, num_students):

        super().__init__()

        self.backbone = models.resnet18(weights='IMAGENET1K_V1')

        feature_dim = self.backbone.fc.in_features

        self.backbone.fc = nn.Identity()

        # Student recognition head
        self.student_head = nn.Linear(feature_dim, num_students)

        # Emotion recognition head
        self.emotion_head = nn.Linear(feature_dim, 3)

    def forward(self, x):

        features = self.backbone(x)

        student_output = self.student_head(features)

        emotion_output = self.emotion_head(features)

        return student_output, emotion_output


# ========================= TRAINING =========================
if __name__ == "__main__":

    dataset = RecogEmotionDataset(transform=transform)

    if len(dataset) == 0:
        raise ValueError("No images found!")

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    num_students = dataset.num_students

    if num_students == 0:
        raise ValueError("No student folders found in recognition/train")

    model = RecogEmotionModel(
        num_students=num_students
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    criterion = nn.CrossEntropyLoss()

    print("\n🚀 Starting Training...")
    print(f"Students: {num_students}")
    print(f"Samples: {len(dataset)}")
    print(f"Epochs: {NUM_EPOCHS}\n")

    for epoch in range(NUM_EPOCHS):

        model.train()

        total_loss = 0

        correct_student = 0
        correct_emotion = 0

        total = 0

        loop = tqdm(dataloader)

        for images, student_labels, emotion_labels in loop:

            images = images.to(device)

            student_labels = student_labels.to(device)

            emotion_labels = emotion_labels.to(device)

            optimizer.zero_grad()

            student_out, emotion_out = model(images)

            student_loss = criterion(student_out, student_labels)

            emotion_loss = criterion(emotion_out, emotion_labels)

            loss = student_loss + emotion_loss

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            total += images.size(0)

            correct_student += (
                student_out.argmax(1) == student_labels
            ).sum().item()

            correct_emotion += (
                emotion_out.argmax(1) == emotion_labels
            ).sum().item()

            loop.set_description(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}]"
            )

        student_acc = 100 * correct_student / total
        emotion_acc = 100 * correct_emotion / total

        avg_loss = total_loss / len(dataloader)

        print(
            f"\nEpoch {epoch+1}"
            f" | Loss: {avg_loss:.4f}"
            f" | Student Acc: {student_acc:.2f}%"
            f" | Emotion Acc: {emotion_acc:.2f}%"
        )

        # Save checkpoint
        os.makedirs("models", exist_ok=True)

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'num_students': num_students,
        }, CHECKPOINT_PATH)

    # ========================= FINAL SAVE =========================
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_students': num_students,
    }, MODEL_SAVE_PATH)

    print("\n✅ Training Completed!")
    print(f"Model saved to: {MODEL_SAVE_PATH}")