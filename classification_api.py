from flask import Flask, request
import math
import torchvision
import torch
import argparse

from model import FurnitureClassifier, furniture_transform

app = Flask(__name__)


@app.post("/classify")
def classify():
    image_bytes = request.get_data()
    image_bytes_tensor = torch.frombuffer(image_bytes, dtype=torch.uint8)
    image = torchvision.io.decode_image(image_bytes_tensor)

    image = furniture_transform(image)

    with torch.no_grad():
        output = model(image.unsqueeze(0))

    probabilities = torch.nn.functional.softmax(output, dim=1).tolist()[0]

    return {
        "class": probabilities.index(max(probabilities)),
        "probabilities": {
            classname: float("{:.3f}".format(probability))
            for classname, probability in enumerate(probabilities)
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="path to the model file",
        action="store",
        default=".model.pt",
    )
    parser.add_argument(
        "--port", action="store", help="port to bind HTTP server to", default="4000"
    )
    parser.add_argument(
        "--num_classes",
        action="store",
        help="number of classes in the data set",
        default=9,
    )
    args = parser.parse_args()

    global model
    model = FurnitureClassifier(args.num_classes)
    model.load_state_dict(torch.load(args.model, map_location=torch.device("cpu")))

    app.run(host="0.0.0.0", port=args.port)
