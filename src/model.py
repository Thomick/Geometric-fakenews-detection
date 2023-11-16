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
from dgl.nn import GATConv, AvgPooling

import os

os.environ["DGLBACKEND"] = "pytorch"


class GCNFN(nn.Module):
    def __init__(self, in_channels, n_hidden, n_classes):
        super(GCNFN, self).__init__()
        #print('n_hidden: ', n_hidden)
        #print('in_channels: ', in_channels)
        self.conv1 = GATConv(in_channels, n_hidden, num_heads=1) #GATConv applies graph attention. num_heads adds a dimension, which we remove by applying squeeze in the forward
        self.conv2 = GATConv(n_hidden, n_hidden, num_heads=1)
        self.fc1 = nn.Linear(n_hidden, int(n_hidden/2)) 
        self.fc2 = nn.Linear(int(n_hidden/2), n_classes) 
        self.selu = nn.SELU() 
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, g, x):
        #print data attributes
        #print(g.ndata)
        with g.local_scope(): # To avoid changes to the original graph
            #print("x input shape: ", x.shape)
            # Layer 1 GatConv with SELU non linearity
            #turn x to float
            x = x.float()
            x = self.selu(self.conv1(g, x).squeeze(1)) # dimension: 32,N,64
            #print("x shape after conv1+selu layer", x.shape)

            # Layer 2 GatConv with SELU non linearity 
            x = self.selu(self.conv2(g, x).squeeze(1)) # 32,N,64
            #print("x shape after conv2+activation selu layer", x.shape)
            # Store the convolved features back into the graph
            g.ndata['h'] = x.float()

            # Mean Pooling
            x = dgl.mean_nodes(g, 'h') # batch_size, 64
            #print("x shape after mean pooling", x.shape)
            # Layer 3 fully connected with SELU non linearity
            x = self.selu(self.fc1(x)) # 32,32
            #print("x shape after fully connected layer 3+activation selu layer", x.shape)
            # Layer 4 fully connected
            x = self.fc2(x) # 2,32
            #print("x shape after fully connected layer 4", x.shape)
            #soft max
            #print("x shape after log softmax", self.log_softmax(x).shape) #1,2
            return self.log_softmax(x).transpose(1,0) 

