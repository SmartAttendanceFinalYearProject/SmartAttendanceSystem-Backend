import torch
import numpy as np
import time
from pathlib import Path
from typing import List, Dict
from .database import students_collection

# ========================= CONFIG =========================
SIMILARITY_THRESHOLD = 0.40   # Adjust between 0.35 - 0.50 (higher = stricter)

# Simple in-memory cache for registered students
_students_cache = None
_cache_expiry = 0.0
CACHE_TTL = 5.0  # seconds

def _get_registered_students() -> List[Dict]:
    global _students_cache, _cache_expiry
    now = time.time()
    if _students_cache is None or now > _cache_expiry:
        try:
            _students_cache = list(students_collection.find({}, {
                "fullName": 1,
                "studentID": 1,
                "faceEmbedding": 1
            }))
            _cache_expiry = now + CACHE_TTL
        except Exception as e:
            print(f"Error fetching students for cache: {e}")
            if _students_cache is not None:
                return _students_cache  # fallback to stale cache on DB error
            return []
    return _students_cache


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings"""
    emb1 = emb1.flatten()
    emb2 = emb2.flatten()
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-6))


def recognize_student(embedding: np.ndarray) -> Dict[str, any]:
    """
    Search MongoDB for closest matching student using embedding similarity
    """
    try:
        embedding = np.array(embedding, dtype=np.float32).flatten()
        
        # Get registered students (using cache to improve performance)
        students = _get_registered_students()

        best_match = None
        best_score = -1.0

        for student in students:
            if "faceEmbedding" not in student:
                continue
                
            db_embedding = np.array(student["faceEmbedding"], dtype=np.float32)
            similarity = cosine_similarity(embedding, db_embedding)

            if similarity > best_score:
                best_score = similarity
                best_match = {
                    "student_id": student.get("studentID", "Unknown"),
                    "full_name": student.get("fullName", "Unknown"),
                    "similarity": round(similarity * 100, 2)
                }

        if best_match and best_score >= SIMILARITY_THRESHOLD:
            return {
                "student_id": best_match["student_id"],
                "full_name": best_match["full_name"],
                "confidence": best_match["similarity"],
                "recognized": True
            }
        else:
            return {
                "student_id": "Unknown",
                "full_name": "Not registered as student",
                "confidence": round(best_score * 100, 2) if best_score > 0 else 0.0,
                "recognized": False
            }

    except Exception as e:
        print(f"Recognition error: {e}")
        return {
            "student_id": "Unknown",
            "full_name": "Not registered as student",
            "confidence": 0.0,
            "recognized": False
        }


def recognize_faces_in_classroom(faces_list: List[dict]) -> List[dict]:
    """
    Takes list of detected faces and returns recognized students from database
    """
    results = []
    for face in faces_list:
        if face.get("embedding") is None:
            continue
            
        emb = np.array(face["embedding"], dtype=np.float32)
        recog = recognize_student(emb)
        
        results.append({
            "bbox": face["bbox"],
            "confidence_detection": face.get("confidence", 0.0),
            "student_id": recog["student_id"],
            "full_name": recog["full_name"],
            "recognition_confidence": recog["confidence"],
            "recognized": recog["recognized"],
            "landmarks": face.get("landmarks"),
            "pose": face.get("pose")
        })
    
    return results