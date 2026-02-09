import requests
import base64
import json
from PIL import Image, ImageDraw
import io

def create_test_image_with_faces():
    """Create a test image with face-like objects"""
    # Create image
    img = Image.new('RGB', (800, 600), color='lightgray')
    draw = ImageDraw.Draw(img)
    
    # Draw oval faces
    faces = [
        (100, 100, 250, 250),   # Face 1
        (350, 150, 500, 300),   # Face 2  
        (600, 80, 750, 230)     # Face 3
    ]
    
    for i, (x1, y1, x2, y2) in enumerate(faces):
        # Face oval
        draw.ellipse([x1, y1, x2, y2], fill='lightblue', outline='darkblue', width=3)
        
        # Eyes
        eye_y = y1 + (y2 - y1) * 0.35
        draw.ellipse([x1 + 40, eye_y, x1 + 80, eye_y + 25], fill='black')
        draw.ellipse([x2 - 80, eye_y, x2 - 40, eye_y + 25], fill='black')
        
        # Mouth
        mouth_y = y1 + (y2 - y1) * 0.65
        draw.arc([x1 + 40, mouth_y, x2 - 40, mouth_y + 30], 0, 180, fill='black', width=3)
    
    return img

def test_api():
    """Test the face detection API"""
    
    # Create test image
    img = create_test_image_with_faces()
    
    # Convert to base64
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Prepare request
    url = "http://localhost:8000/detect"
    payload = {
        "image_base64": f"data:image/jpeg;base64,{img_str}"
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print("🧪 Testing YOLO Face Detection API")
    print("=" * 50)
    
    try:
        # Send request
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success!")
            print(f"Message: {result['message']}")
            print(f"Total faces: {result['total_faces']}")
            
            if result['faces']:
                print("\nDetected faces:")
                for i, face in enumerate(result['faces']):
                    print(f"  Face {i+1}: ({face['x1']}, {face['y1']}) to ({face['x2']}, {face['y2']}) - Confidence: {face['confidence']:.3f}")
            
            if 'model_info' in result:
                print(f"\nModel info: {result['model_info']}")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

def check_health():
    """Check API health"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        print("\n🩺 Health Check:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"❌ Health check failed: {e}")

def check_root():
    """Check root endpoint"""
    try:
        response = requests.get("http://localhost:8000/", timeout=10)
        print("\n🏠 API Info:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"❌ Root check failed: {e}")

if __name__ == "__main__":
    # Check endpoints
    check_root()
    check_health()
    
    # Test detection
    test_api()