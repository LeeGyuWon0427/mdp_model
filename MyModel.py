# from ultralytics import YOLO
# model = YOLO("yolov8n.pt")
# results = model("image.jpg", show=True)
# print(results)
import cv2
from ultralytics import YOLO

print(cv2.__version__)
# Load a model
model = YOLO("yolov8n.pt")  # COCO dataset으로 pretrained된 model을 불러옴

# Use the model
results = model.predict("https://ultralytics.com/images/bus.jpg")

print(results[0])

results2 = model("/content/data/street01.jpg", conf = 0.5, save = True, line_thickness = 1)
cv2.imshow("results", results2[0].plot())

cv2.waitKey(0)
cv2.destroyAllWindows()
