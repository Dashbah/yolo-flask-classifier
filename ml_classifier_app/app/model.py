from ultralytics import YOLO
import os

model_path = os.path.join("models", "best.onnx")
model = YOLO(model_path, task="classify")

def classify_image(image_path):
    results = model.predict(image_path, imgsz=224, save=False)[0]
    probs = results.probs

    if probs is None:
        return {"prediction": "Unknown", "confidence": 0.0}

    class_index = probs.top1
    class_name = results.names[class_index]
    confidence = float(probs.data[class_index])

    return {"prediction": class_name, "confidence": round(confidence, 4)}