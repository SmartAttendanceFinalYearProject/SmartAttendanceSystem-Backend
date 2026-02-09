"# SmartAttendanceSystem-Backend" 
# Face Detection API with YOLOv8

Simple REST API that detects human faces in images using **YOLOv8** (Ultralytics).

Receives base64-encoded images → returns bounding boxes + confidence scores.

## Requirements

- **Python**: 3.10.8 (strongly recommended) or 3.10.x / 3.11.x  
  → Download: https://www.python.org/downloads/release/python-3108/

## Setup

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd <project-folder>

2. **Create and activate a virtual environment**

Windows

```python -m venv venv
venv\Scripts\activate```


Linux / macOS

python3 -m venv venv
source venv/bin/activate

3. **Install dependencies**
pip install --upgrade pip
pip install -r requirements.txt

4. **Run the API**
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload


---

This will render **clean, professional, and readable** on GitHub.  
If you want, next I can add **API usage examples** or **sample request/response** sections.
