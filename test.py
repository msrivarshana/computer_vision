from ultralytics import YOLO

model = YOLO("runs/detect/train-2/weights/best.pt")

model.predict(
    source = 'datasets/images/val',
    save=True,
    conf=0.25

)
print("Prediction complete!")
