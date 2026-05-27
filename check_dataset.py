import os
from pathlib import Path

dataset_path = Path("dataset/raw_data/student_photos")

def count_images():
    total_students = 0
    total_recog = 0
    total_emotion = 0
    total_pose = 0

    for student_dir in sorted(dataset_path.iterdir()):
        if student_dir.is_dir() and student_dir.name.startswith("student_"):
            total_students += 1
            
            recog_count = len(list((student_dir / "recognition").glob("*.jpg")))
            emotion_count = len(list((student_dir / "emotion").glob("*.jpg")))
            pose_count = len(list((student_dir / "pose").glob("*.jpg")))
            
            total_recog += recog_count
            total_emotion += emotion_count
            total_pose += pose_count
            
            print(f"{student_dir.name:15} → Recognition: {recog_count:2} | Emotion: {emotion_count:2} | Pose: {pose_count:2}")

    print("\n" + "="*60)
    print(f"Total Students     : {total_students}")
    print(f"Total Recognition  : {total_recog} images")
    print(f"Total Emotion      : {total_emotion} images")
    print(f"Total Pose         : {total_pose} images")
    print("="*60)

if __name__ == "__main__":
    if dataset_path.exists():
        count_images()
    else:
        print("Dataset path not found!")