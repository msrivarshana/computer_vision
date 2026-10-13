import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")

while True:
    ret ,frame = cap.read()
    if not ret:
        print("could not read the frames")

    img = model(frame,verbose=False)

    result = img[0].plot()

    cv2.imshow("live_object",result)

    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

cv2.destroyAllWindows()