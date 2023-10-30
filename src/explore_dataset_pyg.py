# Corrupted dataset

from torch_geometric.datasets import UPFD, TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from torch_geometric.transforms import ToUndirected

import os
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

graph_index = 1  # Index of the graph to display

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "UPFD")
train_dataset = UPFD(
    root=path,
    name="politifact",
    feature="content",
    split="train",
    transform=ToUndirected(),
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Display statistics about the dataset
print("####################")
print("Dataset: {}".format(train_dataset))
print("Number of graphs: {}".format(len(train_dataset)))
print("Number of classes: {}".format(train_dataset.num_classes))
print("Number of features per node: {}".format(train_dataset.num_node_features))
print("Number of features per edge: {}".format(train_dataset.num_edge_features))
print("####################")

# Compute distribution of classes
print("####################")
print("Distribution of classes")
classes, counts = train_dataset.y.unique(return_counts=True)
for i in range(len(classes)):
    print("Class {}: {}".format(classes[i], counts[i]))
print("####################")

# Compute distribution of number of nodes
print("####################")
print("Distribution of number of nodes")
sns.histplot([train_dataset[i].num_nodes for i in range(len(train_dataset))])
plt.xlabel("Number of nodes")
print("####################")

# Compute distribution of number of edges
print("####################")
print("Distribution of number of edges")
plt.figure()
sns.histplot([[train_dataset[i].num_edges for i in range(len(train_dataset))]])
plt.xlabel("Number of edges")
print("####################")


# Display a graph from the dataset
print("####################")
data = train_dataset[graph_index]
print(f"Graph {graph_index} in the dataset")
print(data)
print("Number of nodes: {}".format(data.num_nodes))
print("Number of edges: {}".format(data.num_edges))
print("Average node degree: {}".format(data.num_edges / data.num_nodes))
print("Contains isolated nodes: {}".format(data.has_isolated_nodes()))
print("Contains self-loops: {}".format(data.has_self_loops()))
print("Is undirected: {}".format(data.is_undirected()))
# Draw the graph with networkx
plt.figure()
nx.draw(to_networkx(data), with_labels=True)
print("####################")


plt.show()
