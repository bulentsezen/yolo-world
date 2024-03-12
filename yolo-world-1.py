from ultralytics import YOLOWorld

# Initialize a YOLO-World model
model = YOLOWorld('yolov8s-world.pt')  # or select yolov8m/l-world.pt for different sizes

# Execute inference with the YOLOv8s-world model on the specified image
results = model.predict('yemek.jpg')

# Show results
results[0].show()