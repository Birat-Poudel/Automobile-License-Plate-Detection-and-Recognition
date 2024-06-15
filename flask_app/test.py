import pytesseract as pt
import cv2

roi = cv2.imread("./static/roi/N2.jpeg")
text = pt.image_to_string(roi)
print(f"Text: {text}")