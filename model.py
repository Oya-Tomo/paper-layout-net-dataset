from ultralytics import YOLO


def load_yolo_model():
    weights_file = "yolov12l-doclaynet.pt"
    model = YOLO(weights_file)
    return model


if __name__ == "__main__":
    model = load_yolo_model()
    print("YOLO model loaded successfully.")
