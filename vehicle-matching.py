# Step 1: Import necessary libraries and install dependencies
!pip install ultralytics opencv-python-headless pytesseract matplotlib

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive
import os
from ultralytics import YOLO
import pytesseract

# Step 2: Mount Google Drive
drive.mount('/content/drive')

# Step 3: Set up the path to the dataset
dataset_path = '/content/drive/MyDrive/Sample_Images'

# Step 4: Load the pre-trained YOLO model
model = YOLO('yolov8n.pt')

# Step 5: Create a simple approved vehicle database
approved_vehicles = {
    'MH12DE1433': True,
    'GJ01HF2343': True,
    'DL3CAB1111': True,
    # Add more approved vehicles as needed
}

# Step 6: Function to detect vehicle
def detect_vehicle(image):
    results = model(image)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if model.names[cls] in ['car', 'motorcycle', 'truck', 'bus']:
                x1, y1, x2, y2 = box.xyxy[0]
                return model.names[cls], (int(x1), int(y1), int(x2), int(y2))
    return None, None

# Step 7: Function to detect color
def detect_color(image, box):
    xmin, ymin, xmax, ymax = box
    vehicle_image = image[ymin:ymax, xmin:xmax]
    avg_color_per_row = np.average(vehicle_image, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    color = 'Unknown'
    if np.argmax(avg_color) == 0:
        color = 'Blue'
    elif np.argmax(avg_color) == 1:
        color = 'Green'
    elif np.argmax(avg_color) == 2:
        color = 'Red'
    return color

# Step 8: Function to detect and read license plate
def detect_and_read_license_plate(image, box):
    xmin, ymin, xmax, ymax = box
    vehicle_image = image[ymin:ymax, xmin:xmax]
    gray = cv2.cvtColor(vehicle_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 10, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            plate_image = vehicle_image[y:y+h, x:x+w]
            text = pytesseract.image_to_string(plate_image, config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            return text.strip()
    return None

# Step 9: Function to check if a vehicle is approved
def is_approved_vehicle(license_plate):
    return approved_vehicles.get(license_plate, False)

# Step 10: Main function to process images and match vehicles
def process_and_match_vehicles(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return

    vehicle_type, box = detect_vehicle(image)
    if vehicle_type:
        print(f"Detected a {vehicle_type}")
        color = detect_color(image, box)
        print(f"Color: {color}")
        plate_number = detect_and_read_license_plate(image, box)
        if plate_number:
            print(f"Detected License Plate: {plate_number}")
            is_approved = is_approved_vehicle(plate_number)
            print(f"Approved: {'Yes' if is_approved else 'No'}")
        else:
            print("License Plate not detected.")

        # Display the original image and the detected vehicle
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(122)
        vehicle_image = image[box[1]:box[3], box[0]:box[2]]
        plt.imshow(cv2.cvtColor(vehicle_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Detected {vehicle_type}')
        plt.axis('off')

        plt.show()
    else:
        print("No vehicle detected.")

# Step 11: Process all images in the dataset
for filename in os.listdir(dataset_path):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(dataset_path, filename)
        print(f"Processing {filename}:")
        process_and_match_vehicles(image_path)
        print("\n")
