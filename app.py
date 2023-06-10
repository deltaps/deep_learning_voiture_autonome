import base64
import io
import time
from collections import deque

import numpy as np
import torch
from PIL import Image
from flask import Flask, jsonify, request, render_template

from flask_socketio import SocketIO

from flash.image import SemanticSegmentation, SemanticSegmentationData
import threading

from torchvision.transforms.functional import to_tensor, resize

app = Flask(__name__)
socketio = SocketIO(app)

app.original_img = io.BytesIO()
app.predicted_img = io.BytesIO()
app.labeled_img = io.BytesIO()
app.message = "Nothing"

IMAGE_QUEUE = deque(maxlen=10)
device = "cuda" if torch.cuda.is_available() else "cpu"

datamodule = SemanticSegmentationData.from_folders(
    train_folder="data/CameraRGB",
    train_target_folder="data/CameraSeg",
    val_split=0.1,
    transform_kwargs=dict(image_size=(256, 256)),
    num_classes=21,
    batch_size=4,
)

app.model = SemanticSegmentation(
    backbone="mobilenetv3_large_100",
    head="fpn",
    num_classes=datamodule.num_classes
).to(device)


def load_model():
    app.model = SemanticSegmentation.load_from_checkpoint("semantic_segmentation_model.pt")
    app.model.eval()


color_dict = {
    0: (0, 0, 0),
    1: (70, 70, 70),
    2: (100, 40, 40),
    3: (55, 90, 80),
    4: (220, 20, 60),
    5: (153, 153, 153),
    6: (157, 234, 50),
    7: (128, 64, 128),
    8: (244, 35, 232),
    9: (107, 142, 35),
    10: (0, 0, 142),
    11: (102, 102, 156),
    12: (220, 220, 0),
    13: (70, 130, 180),
    14: (81, 0, 81),
    15: (150, 100, 100),
    16: (230, 150, 140),
    17: (180, 165, 180),
    18: (250, 170, 30),
    19: (110, 190, 160),
    20: (170, 120, 50),
    21: (45, 60, 150),
    22: (145, 170, 100)
}


@app.route("/", methods=["POST"])
def enqueue_image_pair():
    data = request.data

    if not data:
        return jsonify({"status": "error", "message": "No image data provided."}), 400

    image_bytes, labeled_image_bytes = data.split(b"[CARLA]")

    IMAGE_QUEUE.append([image_bytes, labeled_image_bytes])

    return jsonify({"status": "image receive"}), 200


def preprocess_image(image):

    image = Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB")
    image = to_tensor(image).unsqueeze(0).to(device)
    image = resize(image, (640, 800))

    return image


def transform_labelised_image(segmented_img):
    rgb_array = np.zeros((segmented_img.shape[0], segmented_img.shape[1], 3), dtype=np.uint8)
    for class_label, color in color_dict.items():
        mask = segmented_img == class_label
        rgb_array[mask] = color
    return rgb_array


def process_images_queue():
    while True:

        if len(IMAGE_QUEUE) > 0:
            print("IL Y A UNE IMAGE")
            image_str, labeled_image_str = IMAGE_QUEUE.popleft()

            image = preprocess_image(image_str)

            print("image type : ", type(image))
            print("image shape : ", image.shape)

            msg = "Image processed successfully"
            status = "success"

            prediction = app.model(image)

            print("prediction type : ", type(prediction))
            print("prediction shape : ", prediction.shape)

            segmented = prediction.argmax(dim=1).squeeze().detach().cpu().numpy()

            print("segemented_image type : ", type(segmented))
            print("segemented_image shape : ", segmented.shape)

            segmented = transform_labelised_image(segmented)

            print("segemented_image type : ", type(segmented))
            print("segemented_image shape : ", segmented.shape)

            segmented_pil = io.BytesIO()
            segmented_str = Image.fromarray(segmented.astype(np.uint8))
            segmented_str.save(segmented_pil, format="PNG")

            image_str = io.BytesIO(base64.b64decode(image_str))

            labeled_image_str = io.BytesIO(base64.b64decode(labeled_image_str))

            app.original_img = base64.b64encode(image_str.getvalue()).decode('utf-8')
            app.predicted_img = base64.b64encode(segmented_pil.getvalue()).decode('utf-8')
            app.labeled_img = base64.b64encode(labeled_image_str.getvalue()).decode('utf-8')
            app.message = msg

            is_dangerous = 0 in segmented

            if is_dangerous:
                msg = "Danger detected in the image."
                status = "danger"

            socketio.emit('update_result', {'original_img': app.original_img,
                                            'predicted_img': app.predicted_img,
                                            'labeled_img': app.labeled_img,
                                            'message': app.message})

            pass

        time.sleep(0.1)


@app.route('/result')
def result():
    return render_template('result.html',
                           original_img=app.original_img,
                           predicted_img=app.predicted_img,
                           labeled_img=app.labeled_img,
                           message=app.message)


@app.after_request

def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

if __name__ == "__main__":
    load_model()

    processing_thread = threading.Thread(target=process_images_queue)
    processing_thread.daemon = True
    processing_thread.start()

    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=True)
    socketio.run(app)
