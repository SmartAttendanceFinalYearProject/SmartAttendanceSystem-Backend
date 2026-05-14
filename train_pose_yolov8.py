import os
from ultralytics import YOLO
from pathlib import Path

# ========================= CONFIG =========================
DATA_PATH = Path("/content/drive/MyDrive/student_photos/pose")   # Root pose folder
DATASET_YAML = "/content/pose_dataset.yaml"   # We will create this

MODEL_SAVE_PATH = "/content/drive/MyDrive/models/yolov8_pose_sitting_standing.pt"

# ========================= CREATE DATASET YAML =========================
yaml_content = f"""
path: {DATA_PATH}
train: train
val: test

names:
  0: person

# For pose, we use YOLOv8 Pose keypoints
kpt_shape: [17, 3]   # 17 keypoints (COCO format)
"""

with open(DATASET_YAML, "w") as f:
    f.write(yaml_content)

print("✅ pose_dataset.yaml created")

# ========================= LOAD YOLOv8 POSE MODEL =========================
model = YOLO("yolov8s-pose.pt")   # Good balance between speed and accuracy

# ========================= TRAINING =========================
results = model.train(
    data=DATASET_YAML,
    epochs=100,
    imgsz=640,
    batch=8,
    name="yolov8_pose_sitting_standing",
    patience=20,                    # Early stopping
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
    # Pose specific
    kpt_shape=[17, 3],
    conf=0.25,
    iou=0.7,
)

print(f"✅ Training Completed! Best model saved at:")
print(results.save_dir / "weights/best.pt")