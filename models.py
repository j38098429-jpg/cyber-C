import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool

class GNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GNNLayer, self).__init__(aggr='mean')
        self.linear = nn.Linear(in_channels, out_channels)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return x_j * edge_attr

    def update(self, aggr_out):
        return self.act(self.linear(aggr_out))

class GNNModel(nn.Module):
    def __init__(self, node_features=3):
        super(GNNModel, self).__init__()
        self.layer1 = GNNLayer(node_features, 64)
        self.layer2 = GNNLayer(64, 128)
        self.layer3 = GNNLayer(128, 256)
        self.fc = nn.Linear(256, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.layer1(x, edge_index, edge_attr)
        x = self.layer2(x, edge_index, edge_attr)
        x = self.layer3(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        return self.fc(x)
