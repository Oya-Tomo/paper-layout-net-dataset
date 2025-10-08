from model import load_yolo_model


def train():
    model = load_yolo_model()
    data_path = "dataset/data.yaml"
    model.train(
        data=data_path, epochs=100, imgsz=640, batch=16, name="yolov12l-doclaynet-paper"
    )


if __name__ == "__main__":
    train()
