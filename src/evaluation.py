# Evaluate the model on the test set

import argparse
import os

from dgl.data import FakeNewsDataset
from dgl.dataloading import GraphDataLoader

import torch


def evaluate(dataset, model, loader):
    model.eval()

    correct = 0
    for graph, labels in loader:
        features = dataset.feature[graph.ndata["_ID"]]
        out = model(graph, features)
        pred = out.argmax(
            dim=1
        )  # argmax returns the indices of the maximum values along an axis

        # print("pred shape : ", pred.shape)
        # print("labels shape : ", labels.shape)
        correct += int((pred == labels).sum())

    return correct / len(loader.dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model on the test set")
    parser.add_argument(
        "--dataset",
        type=str,
        default="politifact",
        help="Dataset to use (politifact or gossipcop)",
    )
    parser.add_argument(
        "--model_path", type=str, default="model.pt", help="Path to the model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for the data loader"
    )

    args = parser.parse_args()

    # Load the dataset
    path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "data", "FakeNews"
    )
    dataset = FakeNewsDataset(args.dataset, "content", path)

    dataset = dataset[dataset.test_mask]

    loader = GraphDataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Load the model
    model = torch.load(args.model_path)

    # Evaluate the model
    acc = evaluate(model, loader)
    print("Accuracy: {:.4f}".format(acc))
