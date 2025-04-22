from flask import Blueprint, request, jsonify
import os
import uuid
from app.model import classify_image

routes = Blueprint("routes", __name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@routes.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    image.save(filepath)

    result = classify_image(filepath)

    os.remove(filepath)
    return jsonify(result)
