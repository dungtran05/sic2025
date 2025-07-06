from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

model = YOLO("best.pt")

@app.route('/')
def index():
    return ("api is running")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    img = Image.open(file.stream).convert("RGB")

    results = model.predict(img, save=False, conf=0.25)

    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls]
            detections.append({
                "class_id": cls,
                "class_name": name,
                "confidence": round(conf, 2)
            })
    
    return jsonify({"predictions": detections})
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)