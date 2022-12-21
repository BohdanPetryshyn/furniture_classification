import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import pandas as pd
from skimage import io

from model import FurnitureClassifier, furniture_transform

TEST_RATIO = 0.25


class FurnitureDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = io.imread(row["frame_name"])
        label = torch.tensor(row["label"], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_data",
        action="store",
        help="path to the training data folder. The folder should contain frames and labels csv file",
        default=".training_frames/labels.csv",
    )
    parser.add_argument(
        "--input_model",
        help="path to the model input file",
        action="store",
        default=".model.pt",
    )
    parser.add_argument(
        "--output_model",
        help="path to the model output file",
        action="store",
        default=".model.pt",
    )
    parser.add_argument(
        "--num_classes",
        action="store",
        help="number of classes in the data set",
        type=int,
        default=9,
    )
    parser.add_argument(
        "--epochs",
        help="number of epochs to train for",
        action="store",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--learning_rate",
        help="learning rate for the optimizer",
        action="store",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--test_only",
        help="learning rate for the optimizer",
        action="store",
        type=bool,
        default=False,
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    model = FurnitureClassifier(args.num_classes)
    if os.path.exists(args.input_model):
        model.load_state_dict(torch.load(args.input_model, map_location=device))
    model.to(device)

    data = pd.read_csv(args.training_data)

    train_threshold = int(len(data) * (1 - TEST_RATIO))
    train_data = data.iloc[:train_threshold]
    test_data = data.iloc[train_threshold:]

    train_ds = FurnitureDataset(train_data, transform=furniture_transform)
    test_ds = FurnitureDataset(test_data, transform=furniture_transform)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=True)

    if not args.test_only:
        print("Training model...")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        for epoch in range(args.epochs):
            running_loss = 0.0
            for i, (images, labels) in enumerate(train_dl):
                optimizer.zero_grad()

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                batch_loss = loss.item()
                running_loss += batch_loss
                print(
                    f"Batch {i + 1} - Loss: {batch_loss} Avarage loss: {running_loss / (i + 1)}"
                )

        print(f"Epoch {epoch + 1} - Loss: {running_loss / len(train_dl)}")

        torch.save(model.state_dict(), args.output_model)

    print("Testing model...")

    correct_pred = {classname: 0 for classname in range(args.num_classes)}
    total_pred = {classname: 0 for classname in range(args.num_classes)}

    with torch.no_grad():
        for images, labels in test_dl:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[label.item()] += 1
                total_pred[label.item()] += 1

    for classname, correct_count in correct_pred.items():
        if total_pred[classname] != 0:
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(
                "Accuracy for class {:5s} is: {:.1f} %".format(str(classname), accuracy)
            )

    total_predictions = sum(total_pred.values())
    total_correct_predictions = sum(correct_pred.values())

    print(f"Total accuracy: {total_correct_predictions / total_predictions * 100}")


if __name__ == "__main__":
    main()
