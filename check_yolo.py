import sys
print(f"Python version: {sys.version}")
print()

# Check torch
try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print(f"   Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
except Exception as e:
    print(f"❌ PyTorch import error: {e}")

print()

# Check ultralytics
try:
    from ultralytics import YOLO
    print("✅ Ultralytics imported successfully")
    
    # List available models
    print("\nChecking available YOLO models...")
    
    # Try to load standard YOLOv8 model first (should work)
    try:
        model = YOLO('yolov8n.pt')  # Standard YOLOv8n
        print("✅ Standard YOLOv8n model loaded successfully")
        
        # Test with a dummy image
        import numpy as np
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(dummy, verbose=False)
        print("✅ Model inference works")
        
    except Exception as e:
        print(f"❌ Could not load standard YOLO model: {e}")
        
except Exception as e:
    print(f"❌ Ultralytics import error: {e}")

print()

# Try specific face detection models
print("Trying to load face detection models...")
face_models = [
    'yolov8n-face.pt',
    'yolov8s-face.pt',
    'yolov8m-face.pt',
    'yolov8l-face.pt'
]

for model_name in face_models:
    try:
        print(f"\nTrying {model_name}...")
        model = YOLO(model_name)
        print(f"✅ {model_name} loaded successfully!")
        
        # Test it
        import numpy as np
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(dummy, verbose=False)
        print(f"✅ {model_name} inference works")
        break
        
    except Exception as e:
        print(f"❌ {model_name} failed: {e}")