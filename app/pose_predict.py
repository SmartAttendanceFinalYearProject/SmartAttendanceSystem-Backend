from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load the YOLOv8 pose model
pose_model = YOLO("yolov8s-pose.pt")

def predict_pose(image: Image.Image):
    """Standing / Sitting using YOLOv8 Pose"""
    try:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        results = pose_model(img_cv, conf=0.25, verbose=False)

        if not results or len(results[0].keypoints) == 0:
            return {"pose": "standing", "pose_confidence": 60.0}

        kpts = results[0].keypoints.data[0]

        # Keypoint indices for ankles in COCO dataset
        # 15: left_ankle, 16: right_ankle
        left_ankle_y = float(kpts[15][1])
        right_ankle_y = float(kpts[16][1])
        avg_ankle_y = (left_ankle_y + right_ankle_y) / 2
        image_height = img_cv.shape[0]

        # If ankles are in the lower ~28% of the image, likely sitting
        if avg_ankle_y > image_height * 0.72:
            return {"pose": "sitting", "pose_confidence": 80.0}
        else:
            return {"pose": "standing", "pose_confidence": 80.0}

    except Exception as e:
        print(f"Pose error: {e}")
        return {"pose": "standing", "pose_confidence": 50.0}