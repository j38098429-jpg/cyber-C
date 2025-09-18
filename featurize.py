import torch
from torch_geometric.data import Data
import numpy as np

def build_graph(coords, energy):
    """把原子坐标构建成 PyG 的图对象"""
    n = coords.shape[0]
    edge_index = []
    edge_attr = []

    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            edge_index.append([i, j])
            edge_index.append([j, i])   # 双向边
            edge_attr.append(d)
            edge_attr.append(d)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)
    x = torch.tensor(coords, dtype=torch.float)  # 节点特征 = 原子坐标
    y = torch.tensor([energy], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
