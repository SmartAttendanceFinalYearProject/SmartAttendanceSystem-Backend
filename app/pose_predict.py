import numpy as np
from typing import Dict, Any

def predict_pose(face: Dict[str, Any], image_width: int, image_height: int) -> Dict[str, Any]:
    """
    Reliable rule-based sitting vs standing using face vertical position
    """
    try:
        bbox = face["bbox"]
        x1, y1, x2, y2 = bbox
        face_center_y = (y1 + y2) / 2
        
        # Adjusted threshold - people sitting/crouching have faces lower in the frame
        if face_center_y > image_height * 0.62:   
            pose_label = "sitting"
            confidence = 78.0
        else:
            pose_label = "standing"
            confidence = 82.0
        
        return {
            "pose": pose_label,
            "pose_confidence": confidence
        }

    except Exception as e:
        print(f"Pose prediction error: {e}")
        return {
            "pose": "standing",
            "pose_confidence": 50.0
        }