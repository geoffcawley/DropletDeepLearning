
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import cv2

kerasimg = load_img('smile.bmp')
kerasimgarray = img_to_array(kerasimg)
print(kerasimgarray)

cv2img = cv2.imread('smile.bmp')
print(cv2img)