# Face Detection API

FastAPI backend for detecting human faces in images using **YOLOv8** (Ultralytics).

## Requirements

- **Python 3.10.8** (strongly recommended) or 3.10.x / 3.11.x  
  Do **not** use Python 3.13 or 3.14
- Virtual environment

## Installation

1. Clone the repository
```bash
git clone <your-repo-url>
cd <project-folder>
```

2. **Create and activate virtual environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Run the API**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```