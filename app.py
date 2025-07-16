from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from deepface import DeepFace
from PIL import Image
from io import BytesIO
import os
import cv2
import numpy as np
import pyodbc
from collections import Counter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)
CORS(app)

# Kết nối SQL Server
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=tcp:asdjnu12uh12husa.database.windows.net;"
    "DATABASE=ai_2025;"
    "UID=sqladmin;"
    "PWD=YourPassword@123"
)
cursor = conn.cursor()

# Load YOLO model
model = YOLO("best.pt")

@app.route("/")
def index():
    return "Face Recognition API is running"

def save_face_to_db(img, name):
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_bytes = img_byte_arr.getvalue()
    cursor.execute("INSERT INTO Faces (Name, Image) VALUES (?, ?)", name, img_bytes)
    conn.commit()

def get_all_faces_from_db():
    cursor.execute("SELECT Name, Image FROM Faces")
    results = cursor.fetchall()
    face_data = []
    for name, img_bytes in results:
        try:
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            face_data.append((name, np.array(img)))
        except:
            continue
    return face_data

def detect_face(image):
    results = model.predict(image, conf=0.3, save=False)
    boxes = results[0].boxes
    if not boxes or len(boxes) == 0:
        return None
    xyxy = boxes[0].xyxy[0].tolist()
    x1, y1, x2, y2 = map(int, xyxy)
    return image.crop((x1, y1, x2, y2))

@app.route("/register", methods=["POST"])
def register():
    name = request.form.get("name")
    file = request.files.get("image")
    if not name or not file:
        return jsonify({"error": "Missing name or image"}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
        face = detect_face(image)
        if face:
            save_face_to_db(face, name)
            return jsonify({"message": "Face registered"}), 200
        return jsonify({"error": "No face detected"}), 400
    except Exception as e:
        return jsonify({"error": f"Image processing error: {str(e)}"}), 500

@app.route("/verify_frame", methods=["POST"])
def verify_frame():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image provided"}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
    except:
        return jsonify({"error": "Invalid image"}), 400

    face = detect_face(image)
    detected_names = []

    if face:
        face_np = np.array(face)
        for db_name, db_img_np in get_all_faces_from_db():
            try:
                result = DeepFace.verify(face_np, db_img_np, enforce_detection=False)
                if result["verified"]:
                    detected_names.append(db_name)
                    break
            except:
                continue

    predictions = [{
        "class_name": name,
        "confidence": 1.0
    } for name in detected_names]

    return jsonify({"predictions": predictions}), 200

@app.route("/faces", methods=["GET"])
def list_faces():
    cursor.execute("SELECT Name, COUNT(*) FROM Faces GROUP BY Name")
    rows = cursor.fetchall()
    return jsonify({"faces": [{"name": name, "count": count} for name, count in rows]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
