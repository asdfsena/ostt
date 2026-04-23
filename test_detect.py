from ultralytics.models.yolo import YOLO

# Load the Triton Server model
model = YOLO("http://127.0.0.1:8200/yolo", task="detect")

# Run inference on the server
results = model("photo.avif")
print(results)
