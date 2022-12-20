from flask import Flask, request
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

    output = output.argmax().item()

    return str(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="path to the model file",
        action="store",
        default="model.pt",
    )
    parser.add_argument(
        "--port", action="store", help="port to bind HTTP server to", default="4000"
    )
    args = parser.parse_args()

    global model
    model = FurnitureClassifier()
    model.load_state_dict(torch.load(args.model, map_location=torch.device("cpu")))

    app.run(host="0.0.0.0", port=args.port)
