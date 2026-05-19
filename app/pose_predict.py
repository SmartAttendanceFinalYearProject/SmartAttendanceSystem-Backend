from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load YOLOv8 Pose model
pose_model = YOLO("yolov8s-pose.pt")
print("✅ YOLOv8 Pose Model Loaded Successfully!")


def predict_pose(image: Image.Image):
    """Return standing or sitting using YOLOv8 Pose"""
    try:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        results = pose_model(img_cv, conf=0.25, verbose=False)

        if not results or len(results[0].keypoints) == 0:
            return {"pose": "standing", "pose_confidence": 60.0}

        # Get keypoints of the first detected person
        kpts = results[0].keypoints.data[0]

        # Use ankle y-position to determine sitting vs standing
        left_ankle_y = float(kpts[15][1])   # left ankle
        right_ankle_y = float(kpts[16][1])  # right ankle
        avg_ankle_y = (left_ankle_y + right_ankle_y) / 2
        
        image_height = img_cv.shape[0]

        if avg_ankle_y > image_height * 0.68:   # You can tune this value
            return {"pose": "sitting", "pose_confidence": 78.0}
        else:
            return {"pose": "standing", "pose_confidence": 78.0}

    except Exception as e:
        print(f"Pose prediction error: {e}")
        return {"pose": "standing", "pose_confidence": 50.0}