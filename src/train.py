# Perform experiments on our models

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch

from dgl.data import FakeNewsDataset
from dgl.dataloading import GraphDataLoader

import model, eval
from model import GCNFN
from eval import evaluate


# Train the model for one epoch
def train_one_epoch(model, loader, optimizer, loss_fn):
    losses = []
    for graph, labels in loader:  # Unpack the graph and labels from your data list
        optimizer.zero_grad()
        #print info on graphs
        #for key in graph.ndata:
        #    print(key)

        out = model(graph)  
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)


# Train the model
def train(model, loader, optimizer, loss_fn):
    model.train()
    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss = model.evaluate(val_loader, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            "Epoch {:03d} train_loss: {:.4f} val_loss: {:.4f}".format(
                epoch, train_loss, val_loss
            )
        )
    return train_losses, val_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiments on our models")
    parser.add_argument(
        "--dataset",
        type=str,
        default="politifact",
        help="Dataset to use (politifact or gossipcop)",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train the model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for the data loader"
    )
    parser.add_argument(
        "--model_name", type=str, default="model.pt", help="Name of the model savefile"
    )

    args = parser.parse_args()

    # Load the dataset
    path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "data", "FakeNews"
    )
    dataset = FakeNewsDataset(args.dataset, "content", path)
    # Debugging: Check the first graph in the dataset
    first_graph, _ = dataset[0]  # Unpack the first item into graph and label

    # Now you can access the ndata of the first graph
    for key in first_graph.ndata:
        print(f"Key: {key}, Shape: {first_graph.ndata[key].shape}")
    # Split the dataset

    #transform dataset.train_mask to integers tensor
    train_indices = torch.nonzero(dataset.train_mask).squeeze()
    val_indices = torch.nonzero(dataset.val_mask).squeeze()
    test_indices = torch.nonzero(dataset.test_mask).squeeze()


    train_dataset = [dataset[idx] for idx in train_indices]
    val_dataset = [dataset[idx] for idx in val_indices]
    test_dataset = [dataset[idx] for idx in test_indices]
    
    '''def assign_features(graph, features, indices):
        graph_features = features[indices]  # Slice the features for the nodes in this graph
        graph.ndata['feat'] = graph_features
        return graph

    # Modify dataset creation
    train_dataset = [
        (assign_features(dataset[idx][0], dataset.feature, dataset[idx][0].ndata['_ID']), dataset[idx][1]) 
        for idx in train_indices
    ]
    val_dataset = [
        (assign_features(dataset[idx][0], dataset.feature, dataset[idx][0].ndata['_ID']), dataset[idx][1]) 
        for idx in val_indices
    ]
    test_dataset = [
        (assign_features(dataset[idx][0], dataset.feature, dataset[idx][0].ndata['_ID']), dataset[idx][1]) 
        for idx in test_indices
    ]'''
    # Create the data loaders
    train_loader = GraphDataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = GraphDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = GraphDataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Create the model
    other_args = {'n_hidden': 64}

    #print(dir(dataset))
    model = GCNFN(dataset.feature.shape[1], other_args["n_hidden"], dataset.num_classes)

    # Train the model
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.HingeEmbeddingLoss()

    train_losses, val_losses = train(model, train_loader, optimizer, loss_fn)

    # Save the model
    save_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "models", args.model_name
    )
    torch.save(model, save_path)

    # Plot the losses
    sns.set_style("darkgrid")
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig("losses.png")
    plt.show()

    # Evaluate the model on the test set
    acc = evaluate(model, test_loader)
    print("Accuracy: {:.4f}".format(acc))
