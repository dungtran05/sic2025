from flask import Flask, request, jsonify
from flask_cors import CORS
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
from io import BytesIO
import os
import numpy as np
import pyodbc
import torch
import cv2
from sklearn.metrics.pairwise import cosine_similarity

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)
CORS(app)

# --- Database Connection ---
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=tcp:asdjnu12uh12husa.database.windows.net;"
    "DATABASE=ai_2025;"
    "UID=sqladmin;"
    "PWD=YourPassword@123"
)
cursor = conn.cursor()

# --- Load FaceNet Model ---
facenet = InceptionResnetV1(pretrained='vggface2').eval()

# --- Haar Cascade Classifier ---
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# --- Preprocess Transform ---
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# --- Helper Functions ---
def detect_face(pil_image):
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]  # lấy khuôn mặt đầu tiên
        cropped_face = pil_image.crop((x, y, x + w, y + h))
        return cropped_face
    return None

def get_face_embedding(face_img):
    face_tensor = transform(face_img).unsqueeze(0)
    with torch.no_grad():
        embedding = facenet(face_tensor)
    return embedding.numpy()[0]

def save_face_to_db(img, name):
    embedding = get_face_embedding(img)
    embedding_bytes = embedding.tobytes()
    cursor.execute("INSERT INTO Faces (Name, Embedding) VALUES (?, ?)", name, embedding_bytes)
    conn.commit()

def get_all_embeddings_from_db():
    cursor.execute("SELECT Name, Embedding FROM Faces")
    return [
        (name, np.frombuffer(embedding, dtype=np.float32))
        for name, embedding in cursor.fetchall()
    ]

# --- Routes ---
@app.route("/")
def index():
    return "Face Recognition API (OpenCV + FaceNet) is running"

@app.route("/register", methods=["POST"])
def register():
    name = request.form.get("name")
    file = request.files.get("image")

    if not name or not file:
        return jsonify({"error": "Missing name or image"}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
        face = detect_face(image)
        if not face:
            return jsonify({"error": "No face detected"}), 400

        save_face_to_db(face, name)
        return jsonify({"message": "Face registered successfully"}), 200

    except Exception as e:
        return jsonify({"error": f"Image processing error: {str(e)}"}), 500

@app.route("/verify_frame", methods=["POST"])
def verify_frame():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image provided"}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
        face = detect_face(image)
        if not face:
            return jsonify({"predictions": []}), 200

        input_embedding = get_face_embedding(face).reshape(1, -1)

        for db_name, db_embedding in get_all_embeddings_from_db():
            similarity = cosine_similarity(input_embedding, db_embedding.reshape(1, -1))[0][0]
            if similarity > 0.6:
                return jsonify({
                    "predictions": [{
                        "class_name": db_name,
                        "confidence": float(similarity)
                    }]
                }), 200

        return jsonify({"predictions": []}), 200

    except Exception as e:
        return jsonify({"error": f"Verification error: {str(e)}"}), 500

@app.route("/faces", methods=["GET"])
def list_faces():
    cursor.execute("SELECT Name, COUNT(*) FROM Faces GROUP BY Name")
    return jsonify({
        "faces": [{"name": name, "count": count} for name, count in cursor.fetchall()]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
