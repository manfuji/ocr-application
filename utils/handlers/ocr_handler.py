
import easyocr
import cv2
import numpy as np

# Initialize the OCR reader
reader = easyocr.Reader(['en'])


def perform_ocr(image: np.ndarray):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform OCR on the image
    result = reader.readtext(gray_image)

    # Extract text from the result
    extracted_text = [entry[1] for entry in result]

    return extracted_text
