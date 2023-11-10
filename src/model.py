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
        print(in_channels)
        self.conv1 = GATConv(in_channels, n_hidden, num_heads=1) #GATConv applies graph attention 
        self.conv2 = GATConv(n_hidden, n_hidden, num_heads=1)
        #print("n_hidden: ", n_hidden)
        #print("n_hidden/2: ", n_hidden/2) 
        self.fc1 = nn.Linear(n_hidden, int(n_hidden/2)) 
        self.fc2 = nn.Linear(int(n_hidden/2), n_classes) 
        self.selu = nn.SELU() 

    def forward(self, g):
        #print(g.ndata)
        with g.local_scope(): # To avoid changes to the original graph
            g = dgl.add_self_loop(g) # Add self loops
            x = g.ndata['_ID']  # Access the node features
            print('x.shape', x.shape)
            edge_index = g.edges()

            # Layer 1 GatConv with SELU non linearity
            x = self.selu(self.conv1(g, x)) # dimension: N,64
            # Layer 2 GatConv with SELU non linearity 
            x = self.selu(self.conv2(g, x)) # N,64
            # Mean Pooling
            x = torch.mean(x, batch) # 1,64
            
            # Layer 3 fully connected with SELU non linearity
            x = self.selu(self.fc1(x)) # 1,32
            # Layer 4 fully connected
            x = self.fc2(x) # 1,2
            #soft max
            return self.log_softmax(x, dim=-1) # 1,2

