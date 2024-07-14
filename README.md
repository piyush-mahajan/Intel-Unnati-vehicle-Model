# Vehicle Detection and License Plate Recognition System

## Overview

This project implements a Vehicle Detection and License Plate Recognition system using computer vision techniques and deep learning models. The system is designed to process images of vehicles, detect the vehicles, identify their color, and recognize their license plates. It's particularly useful for automated parking systems, traffic monitoring, and security applications.

## Features

- Vehicle detection using YOLO (You Only Look Once) model
- Vehicle color identification
- License plate detection and recognition
- Approval status checking against a predefined database

## Dependencies

- ultralytics
- opencv-python-headless
- pytesseract
- matplotlib
- numpy

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/yourusername/vehicle-detection-system.git
   cd vehicle-detection-system
   ```

2. Install the required dependencies:

   ```
   pip install ultralytics opencv-python-headless pytesseract matplotlib numpy
   ```

3. Mount your Google Drive (if using Google Colab):

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. Set up the path to your dataset in the `dataset_path` variable.

## Usage

Run the main script:

```python
python vehicle_detection.py
```

The script will process all images in the specified dataset directory and output the results for each image.

## How It Works

1. **Vehicle Detection**: Uses a pre-trained YOLO model to detect vehicles in the image.
2. **Color Detection**: Analyzes the average color of the detected vehicle region.
3. **License Plate Detection and Recognition**:
   - Applies edge detection and contour analysis to find potential license plate regions.
   - Uses Tesseract OCR to read the text from the detected license plate.
4. **Approval Checking**: Checks the recognized license plate against a predefined list of approved vehicles.

## Output

For each processed image, the system outputs:

- The type of vehicle detected
- The color of the vehicle
- The license plate number (if detected)
- Whether the vehicle is approved or not

It also displays two images:

- The original image
- A cropped image of the detected vehicle

## Limitations and Future Improvements

- The color detection is basic and may not be accurate for all lighting conditions.
- License plate recognition accuracy can be improved, especially for different styles of plates.
- The system currently doesn't handle multiple vehicles in a single image.

Future improvements could include:

- Implementing a more robust color detection algorithm
- Training a custom OCR model for license plate recognition
- Adding support for multiple vehicle detection and recognition in a single image
- Implementing real-time processing for video streams

## Dataset :

stanford cars dataset - https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset

## Contributing

Contributions to improve the system are welcome. Please feel free to submit pull requests or open issues to discuss potential improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- YOLO model developers
- OpenCV community
- Tesseract OCR developers
