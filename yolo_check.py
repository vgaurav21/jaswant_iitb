import torch
import requests
from PIL import Image
from io import BytesIO

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Download the image
image_url = "https://ultralytics.com/images/zidane.jpg"
response = requests.get(image_url)
if response.status_code == 200:
    img = Image.open(BytesIO(response.content))  # Open the image from bytes
else:
    print("❌ Failed to download the image")
    exit()

# Run inference
results = model(img)

# Print detected objects
results.print()

# Show detection results
results.show()
