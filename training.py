import os

import torch
from ultralytics import YOLOv10


def main():
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Path to the model and YAML file
    MODEL_PATH = './code/yolov10/yolov10n.pt'
    YAML_PATH = './code/safety_helmet_dataset/data.yaml'

    # Ensure the existence of the model file
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    else:
        print(f"Model file found at: {MODEL_PATH}")

    # Ensure the existence of the YAML file
    if not os.path.exists(YAML_PATH):
        raise FileNotFoundError(f"YAML file not found at: {YAML_PATH}")
    else:
        print(f"YAML file found at: {YAML_PATH}")

    model = YOLOv10(MODEL_PATH)

    # Transfer model to device (GPU or CPU)
    model.to(device)

    # Hyperparameters
    EPOCHS = 20
    IMG_SIZE = 160
    BATCH_SIZE = 4

    # Training process
    model.train(data=YAML_PATH, epochs=EPOCHS,
                batch=BATCH_SIZE, imgsz=IMG_SIZE)


if __name__ == '__main__':
    main()
