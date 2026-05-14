import os
from ultralytics import YOLO
from pathlib import Path

# ========================= CONFIG =========================
# Update these paths according to your setup
DATASET_YAML = "/content/drive/MyDrive/student_photos/dataset/data.yaml"   # Your existing yaml
MODEL_SAVE_PATH = "/content/drive/MyDrive/models/yolov8_pose_sitting_standing"

print(f"Using dataset config: {DATASET_YAML}")

# Verify yaml exists
if not os.path.exists(DATASET_YAML):
    raise FileNotFoundError(f"data.yaml not found at: {DATASET_YAML}")

# ========================= LOAD & TRAIN YOLOv8 POSE =========================
model = YOLO("yolov8s-pose.pt")  # Good balance between speed and accuracy

print("🚀 Starting YOLOv8 Pose Training for Sitting vs Standing...")

results = model.train(
    data=DATASET_YAML,
    epochs=100,
    imgsz=640,
    batch=8,                    # Adjust based on your GPU memory
    name="yolov8_pose_sitting_standing",
    patience=20,                # Early stopping
    save=True,
    project=Path(MODEL_SAVE_PATH).parent,   # Save in models folder
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
    
    # Pose specific
    kpt_shape=[17, 3],          # COCO keypoints format
    conf=0.25,
    iou=0.7,
)

print("\n🎉 Training Completed Successfully!")
print(f"Best model saved at: {results.save_dir / 'weights/best.pt'}")
print(f"Last model saved at: {results.save_dir / 'weights/last.pt'}")