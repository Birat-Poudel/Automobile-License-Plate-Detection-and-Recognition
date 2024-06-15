# Automobile License Plate Detection and Recognition

1. License Plate Detection
    - Deep Learning Model (Inception-ResNet-v2)
2. Text Extraction from License Plate
    - Google Tesseract (Optical Character Recognition)

#### Limitations of PyTesseract

1. Text shouldn't have any kind of skewness.
2. Text shouldn't be of low resolutions.
3. Text shouldn't be cursive.
4. Text shouldn't have any kind of effects.

##### Solutions for all these limitations:

1. Proper Image Preprocessing using OpenCV, scikit-learn, keras.
2. Build own OCR model.