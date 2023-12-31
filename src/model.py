# Models

"""
four-layer Graph CNN with two convolutional layers (64-dimensional output features map in each) and two fully connected layers (producing 32-
and 2-dimensional output features, respectively). 
One head of graph attention [41] was used in every convolutional layer to implement the filters together with mean-pooling.

We used Scaled Exponential Linear Unit (SELU) [13] as non-linearity throughout the entire network.
Hinge loss was employed to train the neural network . No regularization was used with our model."""

import torch
import torch.nn.functional as F
import torch.nn as nn

import dgl
import dgl.function as fn
from dgl.nn import GATConv, AvgPooling, GraphConv, GlobalAttentionPooling

import os

os.environ["DGLBACKEND"] = "pytorch"


# Original GCNFN model
class GCNFN(nn.Module):
    def __init__(self, in_channels, no_features, n_hidden=64, n_classes=2):
        super(GCNFN, self).__init__()
        self.conv1 = GATConv(
            in_channels, n_hidden, num_heads=1
        )  # GATConv applies graph attention. num_heads adds a dimension, which we remove by applying squeeze in the forward
        self.conv2 = GATConv(n_hidden, n_hidden, num_heads=1)
        self.fc1 = nn.Linear(n_hidden, int(n_hidden / 2))
        self.fc2 = nn.Linear(int(n_hidden / 2), n_classes)
        self.selu = nn.SELU()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.no_features = no_features

    def forward(self, g, x):
        with g.local_scope():  # To avoid changes to the original graph
            # Layer 1 GatConv with SELU non linearity
            x = x.float()
            x = self.selu(self.conv1(g, x).squeeze(1))  # dimension: 32,N,64

            # Layer 2 GatConv with SELU non linearity
            x = self.selu(self.conv2(g, x).squeeze(1))  # 32,N,64
            # Store the convolved features back into the graph
            g.ndata["h"] = x.float()

            # Mean Pooling
            x = dgl.mean_nodes(g, "h")  # batch_size, 64
            # Layer 3 fully connected with SELU non linearity
            x = self.selu(self.fc1(x))  # 32,32
            # Layer 4 fully connected
            x = self.fc2(x)  # 2,32
            return self.log_softmax(x)  # .transpose(1,0)


# Simplified model with scalar output and possibility to use mean or attention pooling
class ModifiedGCNFN(nn.Module):
    def __init__(self, in_channels, n_hidden=64):
        super(ModifiedGCNFN, self).__init__()
        self.conv1 = GATConv(in_channels, n_hidden, num_heads=1)
        self.conv2 = GATConv(n_hidden, n_hidden, num_heads=1)
        self.fc1 = nn.Linear(n_hidden, int(n_hidden / 2))
        self.fc2 = nn.Linear(int(n_hidden / 2), 1)
        self.selu = nn.SELU()
        self.pooling_gate = nn.Linear(n_hidden, 1)
        self.pooling = GlobalAttentionPooling(self.pooling_gate)

    def forward(self, g, x):
        with g.local_scope():
            x = x.float()
            x = self.selu(self.conv1(g, x).squeeze(1))
            x = self.selu(self.conv2(g, x).squeeze(1))
            x = self.pooling(g, x)
            x = self.selu(self.fc1(x))
            return F.sigmoid(self.fc2(x))


# Model without graph attention layer and normalization (Can learn using only the graph structure and no features)
class NoAttentionNet(nn.Module):
    def __init__(self, in_channels, n_hidden=64, pooling="attention"):
        super(NoAttentionNet, self).__init__()
        self.conv1 = GraphConv(in_channels, n_hidden, norm="none")
        self.conv2 = GraphConv(n_hidden, n_hidden, norm="none")
        self.fc1 = nn.Linear(n_hidden, int(n_hidden / 2))
        self.fc2 = nn.Linear(int(n_hidden / 2), 1)
        self.selu = nn.SELU()
        if pooling == "mean":
            self.pooling = AvgPooling()
        elif pooling == "attention":
            self.pooling_gate = nn.Linear(n_hidden, 1)
            self.pooling = GlobalAttentionPooling(self.pooling_gate)

    def forward(self, g, x):
        with g.local_scope():
            x = x.float()
            x = self.conv1(g, x)
            x = self.selu(self.conv2(g, x))
            x = self.pooling(g, x)
            x = self.selu(self.fc1(x))
            return F.sigmoid(self.fc2(x))


# Model without graph convolutional layers. Does not make use of the relations between nodes
class NoConvNet(nn.Module):
    def __init__(self, in_channels, n_hidden=64, pooling="attention"):
        super(NoConvNet, self).__init__()
        self.fc1 = nn.Linear(in_channels, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        if pooling == "mean":
            self.pooling = AvgPooling()
        elif pooling == "attention":
            self.pooling_gate = nn.Linear(n_hidden, 1)
            self.pooling = GlobalAttentionPooling(self.pooling_gate)
        self.fc3 = nn.Linear(n_hidden, n_hidden // 2)
        self.fc4 = nn.Linear(n_hidden // 2, 1)
        self.selu = nn.SELU()

    def forward(self, g, x):
        with g.local_scope():
            x = x.float()
            x = self.selu(self.fc1(x))
            x = self.selu(self.fc2(x))
            x = self.pooling(g, x)
            x = self.selu(self.fc3(x))
            return F.sigmoid(self.fc4(x))
