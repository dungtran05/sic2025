from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
from flasgger import Swagger, swag_from

app = Flask(__name__)
CORS(app)
swagger = Swagger(app)

model = YOLO("best.pt")

@app.route('/')
def index():
    return "API is running"

@app.route('/predict', methods=['POST'])
@swag_from({
    'tags': ['YOLO'],
    'consumes': ['multipart/form-data'],
    'parameters': [
        {
            'name': 'image',
            'in': 'formData',
            'type': 'file',
            'required': True,
            'description': 'Ảnh cần nhận diện'
        }
    ],
    'responses': {
        200: {
            'description': 'Danh sách các đối tượng được nhận diện',
            'examples': {
                'application/json': {
                    "predictions": [
                        {
                            "class_id": 0,
                            "class_name": "person",
                            "confidence": 0.95
                        }
                    ]
                }
            }
        },
        400: {
            'description': 'Thiếu ảnh'
        }
    }
})
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
    print(app.url_map)
# This code sets up a Flask application with an endpoint for image prediction using a YOLO model.