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


# Evaluate the model accuracy on the on the dataset accessible through the loader
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


# Evaluate the model rocauc score on the on the dataset accessible through the loader
def evaluate_auc(model, loader):
    model.eval()
    log_labels = []
    log_out = []
    for graph, labels in loader:
        out = model(graph, graph.ndata["feat"])
        log_labels += labels.tolist()
        log_out += [x[-1].item() for x in out]

    auc = roc_auc_score(log_labels, log_out)
    return auc
