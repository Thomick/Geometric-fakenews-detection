# Models

'''
four-layer Graph CNN with two convolutional layers (64-dimensional output features map in each) and two fully connected layers (producing 32-
and 2-dimensional output features, respectively). 
One head of graph attention [41] was used in every convolutional layer to implement the filters together with mean-pooling.

We used Scaled Exponential Linear Unit (SELU) [13] as non-linearity throughout the entire network.
Hinge loss was employed to train the neural network . No regularization was used with our model.'''

import torch
import torch.nn.functional as F
import torch.nn as nn

import dgl
import dgl.function as fn
from dgl.nn import GATConv

import os

os.environ["DGLBACKEND"] = "pytorch"


class GCNFN(nn.Module):
    def __init__(self, in_channels, n_hidden, n_classes):
        super(GCNFN, self).__init__()
        self.conv1 = GATConv(in_channels, n_hidden, num_heads=1) #GATConv applies graph attention 
        self.conv2 = GATConv(n_hidden, n_hidden, num_heads=1) 
        self.fc1 = nn.Linear(n_hidden, n_hidden) 
        self.fc2 = nn.Linear(n_hidden, n_classes) 
        self.selu = nn.SELU() 

    def forward(self, g):
        #print(g.ndata)
        g = g.local_scope()  # To avoid changes to the original graph
        x = g.ndata['_ID']  # Access the node features
        edge_index = g.edges()

        # Layer 1 GatConv with SELU non linearity
        x = self.selu(self.conv1(x, edge_index)) # dimension: N,64
        # Layer 2 GatConv with SELU non linearity 
        x = self.selu(self.conv2(x, edge_index)) # N,64
        # Mean Pooling
        x = torch.mean(x, batch) # 1,64
        
        # Layer 3 fully connected with SELU non linearity
        x = self.selu(self.fc1(x)) # 1,32
        # Layer 4 fully connected
        x = self.fc2(x, edge_index) # 1,2
        #soft max
        return self.log_softmax(x, dim=-1) # 1,2

