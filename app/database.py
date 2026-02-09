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

# Collection for registered users + embeddings
users_collection = db["users"]