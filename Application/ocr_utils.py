import pytesseract
from PIL import Image
import cv2
import numpy as np

# Set this path according to your system
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def preprocess_image(image):
    """Improve OCR accuracy"""
    img = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise removal
    gray = cv2.medianBlur(gray, 3)

    # Thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    return thresh


def extract_text_from_image(image_file):
    image = Image.open(image_file)

    processed = preprocess_image(image)

    text = pytesseract.image_to_string(processed)

    return text.strip()
