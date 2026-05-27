import os
from ultralytics import YOLO
from pathlib import Path

# ========================= CONFIG =========================
DATASET_YAML = "/content/drive/MyDrive/student_photos/dataset/data.yaml"
PROJECT_NAME = "yolov8_pose_sitting_standing"
SAVE_DIR = "/content/drive/MyDrive/models"

print(f"Using dataset config: {DATASET_YAML}")

# Check if yaml exists
if not os.path.exists(DATASET_YAML):
    raise FileNotFoundError(f"data.yaml not found at: {DATASET_YAML}")

# ========================= TRAIN YOLOv8 POSE =========================
model = YOLO("yolov8s-pose.pt")   # Best balance for pose

print("🚀 Starting YOLOv8 Pose Training...")

results = model.train(
    data=DATASET_YAML,
    epochs=100,
    imgsz=640,
    batch=8,                    # Reduce to 4 if you get CUDA OOM
    name=PROJECT_NAME,
    patience=20,
    save=True,
    project=SAVE_DIR,
    exist_ok=True,
    pretrained=True,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,

    # Augmentations
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=0.5,
    
    # Pose related (these are valid)
    pose=12.0,      # Pose loss weight
    kobj=2.0,       # Keypoint objectness loss weight
    conf=0.25,
    iou=0.7,
)

print("\n🎉 Training Completed!")
print(f"Best model saved at: {results.save_dir / 'weights/best.pt'}")
print(f"Last model saved at: {results.save_dir / 'weights/last.pt'}")