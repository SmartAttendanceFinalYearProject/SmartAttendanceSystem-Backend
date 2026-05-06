import numpy as np
from typing import Dict, Any

def predict_pose(face: Dict[str, Any], image_width: int, image_height: int) -> Dict[str, Any]:
    """
    Simple but effective pose detection for group photos
    - Uses vertical position to detect sitting vs standing
    """
    try:
        bbox = face["bbox"]
        x1, y1, x2, y2 = bbox
        face_center_y = (y1 + y2) / 2
        
        # Rule: If face is in lower part of image → more likely sitting
        if face_center_y > image_height * 0.58:   # You can adjust this value (0.55 - 0.65)
            pose_label = "sitting"
        else:
            pose_label = "standing"
        
        # Optional: Use InsightFace angles if available
        if face.get("pose"):
            pitch = face["pose"].get("pitch", 0)
            if abs(pitch) > 25:
                head_direction = "looking_up" if pitch < 0 else "looking_down"
            else:
                head_direction = "front"
        else:
            head_direction = "front"
        
        return {
            "pose": pose_label,
            "pose_confidence": 70.0,      # Fixed confidence for rule-based method
            "head_direction": head_direction
        }

    except Exception as e:
        print(f"Pose prediction error: {e}")
        return {
            "pose": "standing",
            "pose_confidence": 50.0,
            "head_direction": "front"
        }