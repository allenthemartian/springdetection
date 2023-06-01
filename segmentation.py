import torch
from PIL import Image
import io
from ultralytics import YOLO

def get_yolov5():
    model = YOLO("./ultralytics/yolo/best.onnx")
    model.conf = 0.25
    return model


def get_image_from_bytes(binary_image, max_size=960): #2560
    input_image = Image.open(io.BytesIO(binary_image))
    return input_image
