# Evaluate the model on the test set

import argparse
import os

from dgl.data import FakeNewsDataset
from dgl.dataloading import GraphDataLoader

import torch

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    roc_auc_score,
    average_precision_score,
)


def evaluate(model, loader):
    model.eval()
    dataset_size = len(loader.dataset)
    accuracy = 0
    for graph, labels in loader:
        size = labels.shape[0]
        features = graph.ndata["feat"]
        out = model(graph, features)
        labels = labels.type(torch.LongTensor)
        if out.shape[-1] == 1:
            pred = (out > 0.5).long().squeeze()
        else:
            pred = out.argmax(dim=1)

        accuracy += accuracy_score(labels, pred) * size

    return accuracy / dataset_size


def evaluate_auc(model, loader):
    model.eval()
    log_labels = []
    log_out = []
    for graph, labels in loader:
        out = model(graph, graph.ndata["feat"])
        log_labels += labels.tolist()
        log_out += [x.item() for x in out]

    auc = roc_auc_score(log_labels, log_out)
    return auc


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
    """
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
    acc = evaluate(model, loader)[0]
    print("Accuracy: {:.4f}".format(acc))"""
