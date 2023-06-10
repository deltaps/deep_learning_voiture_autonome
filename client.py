import base64
import time
from io import BytesIO

import requests
from PIL import Image

SERVER_URL = "http://127.0.0.1:5000/"


def send_images(image, labeled_image):
    data = image + b"[CARLA]" + labeled_image

    response = requests.post(SERVER_URL, data=data)

    if response.status_code == 200:
        print("Images sending successfully.")
    else:
        print("Error sending images to server.")


def prepare_image(image_dir, labeled_image_dir):
    image = Image.open(image_dir)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())

    labeled_image = Image.open(labeled_image_dir)

    buffered = BytesIO()
    labeled_image.save(buffered, format="PNG")
    img_labeled_str = base64.b64encode(buffered.getvalue())

    return img_str, img_labeled_str


if __name__ == '__main__':
    for i in range(1, 9):
        image_dir = f"./data/CameraRGB/F6{i}-{i}.png"
        image_lab_dir = f"./data/CameraSeg/F6{i}-{i}.png"

        image_str, labeled_image_str = prepare_image(image_dir, image_lab_dir)

        send_images(image_str, labeled_image_str)

        time.sleep(3)


