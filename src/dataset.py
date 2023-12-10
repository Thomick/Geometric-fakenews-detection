import os
import torch
from dgl.data import FakeNewsDataset
import dgl
from dgl.dataloading import GraphDataLoader


class DatasetManager:
    def __init__(self, dataset_name, features_name, no_features, batch_size):
        self.path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "data", "FakeNews"
        )
        self.dataset = FakeNewsDataset(dataset_name, features_name, self.path)
        self.dataset.process()
        self.batch_size = batch_size
        self.no_features = no_features

    def get_train_loader(self):
        train_indices = torch.nonzero(self.dataset.train_mask).squeeze()
        train_dataset = [
            (
                self.assign_features(self.dataset[idx][0], self.dataset.feature),
                self.dataset[idx][1],
            )
            for idx in train_indices
        ]
        train_loader = GraphDataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        return train_loader

    def get_val_loader(self):
        val_indices = torch.nonzero(self.dataset.val_mask).squeeze()
        val_dataset = [
            (
                self.assign_features(self.dataset[idx][0], self.dataset.feature),
                self.dataset[idx][1],
            )
            for idx in val_indices
        ]
        val_loader = GraphDataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        return val_loader

    def get_test_loader(self):
        test_indices = torch.nonzero(self.dataset.test_mask).squeeze()
        test_dataset = [
            (
                self.assign_features(self.dataset[idx][0], self.dataset.feature),
                self.dataset[idx][1],
            )
            for idx in test_indices
        ]
        test_loader = GraphDataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
        return test_loader

    def assign_features(self, graph, features):
        if self.no_features:
            graph_features = torch.ones((graph.ndata["_ID"].shape[0], 1))
        else:
            graph_features = features[graph.ndata["_ID"]]
        graph.ndata["feat"] = graph_features
        graph = dgl.add_self_loop(graph)
        return graph

    def get_dataset(self):
        return self.dataset

    def get_feature(self):
        return self.dataset.feature

    def get_labels(self):
        return self.dataset.labels

    def get_num_features(self):
        if self.no_features:
            return 1
        else:
            return self.dataset.feature.shape[1]
