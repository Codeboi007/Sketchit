import os
import cv2
import numpy as np
from flask import Flask, render_template, request, send_file

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads/"
SKETCH_FOLDER = "static/sketches/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SKETCH_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Convert to B/W sketch
            sketch_path = convert_to_sketch(filepath, file.filename)

            return render_template("index.html", sketch=sketch_path, uploaded=filepath)

    return render_template("index.html", sketch=None, uploaded=None)

def convert_to_sketch(image_path, filename):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    inverted = cv2.bitwise_not(gray)  # Invert image
    blurred = cv2.GaussianBlur(inverted, (21, 21), sigmaX=0, sigmaY=0)
    inverted_blurred = cv2.bitwise_not(blurred)
    sketch = cv2.divide(gray, inverted_blurred, scale=256.0)

    sketch_path = os.path.join(SKETCH_FOLDER, filename)
    cv2.imwrite(sketch_path, sketch)

    return sketch_path

@app.route("/download/<filename>")
def download(filename):
    return send_file(os.path.join(SKETCH_FOLDER, filename), as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
