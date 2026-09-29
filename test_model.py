import ultralytics
from ultralytics import YOLO
import cv2
import numpy

model = YOLO("best.pt") 
image_path = r"C:\Users\aidan\Documents\markerless_tracking\dataset\images\test\frame_27.jpg"

test_image = cv2.imread(image_path)

results = model(test_image, imgsz=640, conf=0.5, verbose=False)