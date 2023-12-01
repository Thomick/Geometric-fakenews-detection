# Perform experiments on our models

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch
import dgl
import tqdm

from dgl.data import FakeNewsDataset
from dgl.dataloading import GraphDataLoader

import model, evaluation
from dataset import DatasetManager
from model import GCNFN, ModifiedGCNFN, NoConvNet
from evaluation import evaluate, evaluate_auc


# Train the model for one epoch
def train_one_epoch(model, train_loader, optimizer, loss_fn):
    losses = []
    for (
        graph,
        labels,
    ) in train_loader:  # Unpack the graph and labels from your data list
        optimizer.zero_grad()
        # features = dataset.feature[graph.ndata['_ID']]
        x = graph.ndata["feat"]
        out = model(graph, x)
        # print values and shape of out
        # print(out.shape)
        # print(out)
        # turn labels into -1 and 1
        # labels = torch.where(labels == 0, -1, labels)

        if out.shape[-1] == 1:
            out = out.squeeze()
            loss = torch.nn.BCELoss()(out, labels.float())
        else:
            labels = labels.type(torch.LongTensor)
            loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)


# Train the model
def train(model, loader, optimizer, loss_fn, val_loader=None):
    model.train()
    train_losses = []
    val_accs = []
    train_accs = []
    # Define tdqm progress bar
    disable = False  # False
    pbar = tqdm.tqdm(range(args.epochs), disable=disable)

    for epoch in range(args.epochs):  # 100 by default
        train_loss = train_one_epoch(model, loader, optimizer, loss_fn)
        val_acc = evaluate(model, val_loader) if val_loader else None
        train_acc = evaluate(model, loader)

        train_losses.append(train_loss)
        val_accs.append(val_acc)
        train_accs.append(train_acc)

        # Update the progress bar with the latest metrics
        pbar.set_postfix(
            train_loss=train_loss, val_acc=val_acc, refresh=True, epoch=epoch
        )
        pbar.update()
    return train_losses, train_accs, val_accs


if __name__ == "__main__":
    default_epochs = 200
    default_dataset = "politifact"
    default_features = "content"

    parser = argparse.ArgumentParser(description="Experiments on our models")
    parser.add_argument(
        "--dataset",
        type=str,
        default=default_dataset,
        help="Dataset to use (politifact or gossipcop)",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=default_features,
        help="Features to use (content, bert, spacy, profile)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=default_epochs,
        help="Number of epochs to train the model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for the data loader"
    )
    parser.add_argument(
        "--model_name", type=str, default="model.pt", help="Name of the model savefile"
    )

    args = parser.parse_args()
    print(args.features)
    dm = DatasetManager(args.dataset, args.features, args.batch_size)

    # Create the data loaders
    train_loader = dm.get_train_loader()
    val_loader = dm.get_val_loader()
    test_loader = dm.get_test_loader()
    print("Number of training samples: ", dm.dataset.train_mask.sum().item())
    print("Number of validation samples: ", dm.dataset.val_mask.sum().item())
    print("Number of test samples: ", dm.dataset.test_mask.sum().item())
    print("Number of features: ", dm.get_num_features())

    # Create the model

    # print(dir(dataset))
    # dm.get_num_features renvoie le nombre de features
    model = GCNFN(dm.get_num_features())
    # model = ModifiedGCNFN(dm.get_num_features())
    # model = NoConvNet(dm.get_num_features())

    # Train the model
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.NLLLoss()
    # torch.nn.CrossEntropyLoss()
    # loss_fn = torch.nn.HingeEmbeddingLoss()

    train_losses, train_accs, val_accs = train(
        model, train_loader, optimizer, loss_fn, val_loader
    )

    # Save the model
    save_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "models", args.model_name
    )
    torch.save(model, save_path)

    # Plot the losses
    sns.set_style("darkgrid")
    plt.plot(train_losses, label="Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig("losses.png")

    plt.figure()
    test_acc = evaluate(model, test_loader)
    print("Train accuracy: {:.4f}".format(train_accs[-1]))
    print("Validation accuracy: {:.4f}".format(val_accs[-1]))
    print("Test accuracy: {:.4f}".format(test_acc))
    plt.plot(val_accs, label="Validation accuracy")
    plt.plot(train_accs, label="Train accuracy")
    plt.axhline(
        y=test_acc, color="r", linestyle="-", label="Test accuracy(after training)"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig("accuracy.png")
    plt.show()

    # Evaluate the model on the test set
    acc = evaluate(model, test_loader)
    print("Accuracy: {:.4f}".format(acc))
    print("ROCAUC: {:.4f}".format(evaluate_auc(model, test_loader)))
