from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8m-seg.pt")

img = cv2.imread("6.jpg")
img = cv2.resize(img, (640, 640))

results = model(img)
result = results[0]

masks = result.masks.data.cpu().numpy()
print("masks datastype:", masks.dtype)
print("masks shape:", masks.shape)

mask = (np.sum(masks, axis=0)).astype(np.float32)
mask = np.clip(mask, 0, 1).astype(np.float32)

mask = cv2.GaussianBlur(mask, (25, 25), 0)

masked_img = img.astype(np.float32) * mask[:, :, None]
masked_img = np.clip(masked_img, 0, 255)
masked_img = masked_img.astype(np.uint8)

mask = (mask * 255).astype(np.uint8)

cv2.imshow("Mask", mask)
cv2.imshow("Masked", masked_img)
cv2.waitKey(0)
cv2.destroyAllWindows()