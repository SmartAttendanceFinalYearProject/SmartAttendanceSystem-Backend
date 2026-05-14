import os
from ultralytics import YOLO
from pathlib import Path

# ========================= CONFIG =========================
DATA_PATH = Path("/content/drive/MyDrive/student_photos/pose")
DATASET_YAML = "/content/pose_dataset.yaml"
MODEL_SAVE_PATH = "/content/drive/MyDrive/models/yolov8_pose_sitting_standing.pt"

# ========================= CREATE DATASET YAML =========================
yaml_content = f"""
path: {DATA_PATH}
train: train
val: test

names:
  0: person

# YOLOv8 Pose settings
kpt_shape: [17, 3]   # 17 keypoints (COCO format)
"""

with open(DATASET_YAML, "w") as f:
    f.write(yaml_content.strip())

print("✅ Dataset YAML created successfully!")

# ========================= TRAIN YOLOv8 POSE =========================
model = YOLO("yolov8s-pose.pt")   # Best balance

print("🚀 Starting YOLOv8 Pose Training...")

results = model.train(
    data=DATASET_YAML,
    epochs=100,
    imgsz=640,
    batch=8,
    name="yolov8_pose_sitting_standing",
    patience=20,
    save=True,
    project="/content/drive/MyDrive/models",
    exist_ok=True,
    pretrained=True,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    conf=0.25,
    iou=0.7,
    # Pose specific
    kpt_shape=[17, 3],
    degrees=10.0,           # augmentation
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
)

print("\n🎉 Training Completed!")
print("Best model saved at:")
print(results.save_dir / "weights/best.pt")