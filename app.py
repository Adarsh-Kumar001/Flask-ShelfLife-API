from flask import Flask, request, jsonify
import torch
from PIL import Image
import os
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
model_path = "best_yolov8_shelf_life.pt"
model = YOLO(model_path)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Updated estimated shelf life in days for fresh vegetables/fruits
SHELF_LIFE = {
    "fresh apple": 30, "fresh banana": 7, "fresh bellpepper": 10,
    "fresh carrot": 20, "fresh cucumber": 7, "fresh mango": 7,
    "fresh orange": 20, "fresh potato": 60
}

@app.route("/predict", methods=["POST"])
def predict():
    print("Incoming request...")

    # Check if 'file' is in request
    if "file" not in request.files:
        print("No 'file' key in request.files")  # Debug
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    
    if file.filename == "":
        print("Empty filename")  # Debug
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)
    print(f"File saved at {image_path}")

    try:
        # Load image
        image = Image.open(image_path)

        # Perform inference
        results = model(image)

        # Parse results
        detections = []
        for result in results:
            for box in result.boxes:
                class_name = result.names[int(box.cls)]
                confidence = float(box.conf)
                bbox = box.xyxy.tolist()[0]

                # Determine if rotten or fresh
                if "rotten" in class_name:
                    message = f"⚠️ Remove the {class_name.replace('rotten ', '')} from the shelf immediately!"
                else:
                    estimated_days = SHELF_LIFE.get(class_name, "Unknown")
                    message = f"✅ Estimated shelf life: {estimated_days} days."

                detections.append({
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": bbox,
                    "message": message
                })

        return jsonify({"detections": detections})
    except Exception as e:
        print(f"Error: {e}")  # Debug
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
