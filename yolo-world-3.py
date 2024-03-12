from ultralytics import YOLO

# Initialize a YOLO-World model
model = YOLO('yolov8s-world.pt')  # or choose yolov8m/l-world.pt

# Define custom classes
model.set_classes(["bean", "rice"])

# Execute prediction for specified categories on an image
results = model.predict('yemek2.jpg')

# Show results
results[0].show()