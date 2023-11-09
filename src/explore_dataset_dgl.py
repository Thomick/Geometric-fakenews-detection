import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from dgl.data import FakeNewsDataset

graph_index = 1  # Index of the graph to display

path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "data", "FakeNews"
)

dataset = FakeNewsDataset("politifact", "content", path)

# Display statistics about the dataset
print("########## DATASET ##########")
print(dataset)
print("Number of graphs: ", dataset.num_graphs)
print("Number of classes: ", dataset.num_classes)
print("Number of features per node: ", dataset.feature.shape[1])
print("Train split: ", dataset.train_mask.sum().item())
print("Validation split: ", dataset.val_mask.sum().item())
print("Test split: ", dataset.test_mask.sum().item())

# Print distribution of labels
labels = dataset.labels
print("########## LABELS ##########")
print("Number of true news: ", torch.sum(labels == 0, dtype=int))
print("Number of fake news: ", torch.sum(labels == 1, dtype=int))

# Print distribution of number of nodes
num_nodes = []
for i in range(len(dataset)):
    graph = dataset[i][0]
    num_nodes.append(graph.number_of_nodes())
print("########## NUMBER OF NODES ##########")
print("Min: ", min(num_nodes))
print("Max: ", max(num_nodes))
print("Mean: ", sum(num_nodes) / len(num_nodes))
print("Median: ", sorted(num_nodes)[len(num_nodes) // 2])
sns.histplot(num_nodes)
plt.xlabel("Number of nodes")

# Print distribution of number of edges
num_edges = []
for i in range(len(dataset)):
    graph = dataset[i][0]
    num_edges.append(graph.number_of_edges())
print("########## NUMBER OF EDGES ##########")
print("Min: ", min(num_edges))
print("Max: ", max(num_edges))
print("Mean: ", sum(num_edges) / len(num_edges))
print("Median: ", sorted(num_edges)[len(num_edges) // 2])
plt.figure()
sns.histplot(num_edges)
plt.xlabel("Number of edges")

# Print degree distribution
degrees = []
for i in range(len(dataset)):
    graph = dataset[i][0]
    degrees.extend(graph.out_degrees().tolist())
print("########## OUT DEGREE DISTRIBUTION ##########")
print("Min: ", min(degrees))
print("Max: ", max(degrees))
print("Mean: ", sum(degrees) / len(degrees))
print("Median: ", sorted(degrees)[len(degrees) // 2])
plt.figure()
sns.histplot(degrees, log_scale=(False, True))
plt.xlabel("Out degree")


# Print information about a graph
print("########## GRAPH ##########")
graph = dataset[graph_index][0]
print(graph)
print("Number of nodes: ", graph.number_of_nodes())
print("Number of edges: ", graph.number_of_edges())
print("Node types: ", graph.ntypes)
print("Edge types: ", graph.etypes)

# Draw the graph
nx_graph = graph.to_networkx()
plt.figure()
nx.draw(nx_graph, with_labels=True)
#plt.show()
