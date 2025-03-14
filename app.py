from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()

# Load the trained model
model = YOLO("best_yolov8_shelf_life.pt")  

# Define shelf life for fresh items
shelf_life = {
    "fresh apple": 30, "fresh banana": 7, "fresh bellpepper": 10,
    "fresh carrot": 20, "fresh cucumber": 7, "fresh mango": 7,
    "fresh orange": 20, "fresh potato": 60
}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    image_np = np.array(image)

    # Run YOLO detection
    results = model(image_np)

    detections = []
    warnings = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0]) * 100  # Convert to percentage
            class_name = model.names[class_id]

            if "rotten" in class_name.lower():
                warnings.append(f"⚠ Remove {class_name} from the shelf!")
            elif class_name in shelf_life:
                detections.append({
                    "item": class_name,
                    "confidence": round(confidence, 2),
                    "shelf_life": shelf_life[class_name]
                })

    return {"detections": detections, "warnings": warnings}
