from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL")
MONGODB_DB_NAME = os.getenv("MONGODB_DB", "attendance_db")

if not MONGODB_URL:
    raise ValueError("MONGODB_URL not set in .env file")

client = MongoClient(MONGODB_URL)
db = client[MONGODB_DB_NAME]

# ====================== COLLECTIONS ======================
students_collection = db["students"]
subjects_collection = db["subjects"]
teachers_collection = db["teachers"]
classes_collection = db["classes"]
attendance_collection = db["attendance_records"]

# Create indexes for better performance
students_collection.create_index("studentID", unique=True)
teachers_collection.create_index("username", unique=True)
classes_collection.create_index("class_name")