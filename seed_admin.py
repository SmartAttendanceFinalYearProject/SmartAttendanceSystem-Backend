from app.database import teachers_collection
from app.auth import get_password_hash
from datetime import datetime

def create_default_admin():
    if teachers_collection.find_one({"username": "admin"}):
        print("✅ Default admin already exists")
        return

    admin = {
        "full_name": "System Administrator",
        "username": "admin",
        "password": get_password_hash("admin123"),   # Change this after first login!
        "role": "admin",
        "created_at": datetime.utcnow()
    }

    teachers_collection.insert_one(admin)
    print("✅ Default Admin created successfully!")
    print("Username: admin")
    print("Password: admin123")
    print("⚠️  Please change the password immediately after first login!")


if __name__ == "__main__":
    create_default_admin()