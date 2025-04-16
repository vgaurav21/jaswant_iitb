from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import os
import numpy as np
from werkzeug.utils import secure_filename
import torch
from PIL import Image

app = Flask(__name__)

# Verify YOLOv5 model loading
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    print("✅ YOLOv5 Loaded Successfully!")
except Exception as e:
    print("❌ Error Loading YOLOv5:", e)
    exit()

# Set upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize video capture
cap = cv2.VideoCapture(0)

def detect_objects(image_path):
    img = Image.open(image_path)
    results = model(img)
    results.print()  # Print detection results in the terminal
    results.save(save_dir=UPLOAD_FOLDER)  # Save detected image with bounding boxes
    
    # Check if any detections were made
    if results.xyxy[0].shape[0] == 0:
        print("No objects detected.")
        return None  # No detections
    
    detected_image_path = os.path.join(UPLOAD_FOLDER, os.path.basename(image_path))
    return detected_image_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        detected_image_path = detect_objects(file_path)
        return render_template('index.html', uploaded_image=detected_image_path, no_detection=(detected_image_path is None))

# Live video feed with object detection
@app.route('/video_feed')
def video_feed():
    def generate_frames():
        while True:
            success, frame = cap.read()
            if not success:
                break
            else:
                # Perform object detection on each frame
                results = model(frame)
                for result in results.xyxy[0]:
                    x1, y1, x2, y2, conf, cls = result.tolist()
                    label = f"{model.names[int(cls)]} {conf:.2f}"
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
