import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import cv2
import pytesseract as pt

model = load_model('./static/models/object_detection_final.h5')

def object_detection(image_path, filename):
  image = cv2.imread(image_path)
  image_array = np.array(image, dtype=np.uint8)
  h, w, d = image_array.shape

  image_test = load_img(image_path, target_size=(224,224))
  image_test_arr = img_to_array(image_test) / 255.0
  image_test_arr_reshaped = image_test_arr.reshape(1,224,224,3)

  coords = model.predict(image_test_arr_reshaped)

  denorm = np.array([w,w,h,h])
  coords = coords * denorm
  coords = coords.astype(np.int32)

  xmin, xmax, ymin, ymax = coords[0]

  cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,255,0), 3)
  
  # image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  
  cv2.imwrite(f"./static/predict/{filename}", image)
  
  return coords

def ocr(image_path, filename):
    img = np.array(load_img(image_path))
    coords = object_detection(image_path, filename)
    xmin, xmax, ymin, ymax = coords[0]
    roi = img[ymin:ymax, xmin:xmax]
    text = pt.image_to_string(roi)
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"./static/roi/{filename}", roi_bgr)
    print(f"Text: {text}")
    return text