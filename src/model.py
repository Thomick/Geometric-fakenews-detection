# Models
import torch
import torch.nn.functional as F
import torch.nn as nn

import dgl
import dgl.function as fn
from dgl.nn import GATConv

import os

os.environ["DGLBACKEND"] = "pytorch"


class GCNFN(nn.Module):
    pass
