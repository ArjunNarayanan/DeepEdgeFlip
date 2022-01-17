import pandas as pd
import torch
from torch_geometric.utils import to_undirected
import edgeflip
import numpy as np

connectivity = pd.read_csv("ideal-mesh\\connectivity.csv", header=None)
vertex_degree = pd.read_csv("ideal-mesh\\degree.csv", header=None)

edges = torch.tensor(connectivity.values, dtype=torch.long).t()
edge_index = to_undirected(edges)
num_edges = edge_index.shape[1]

count_flips = 0
for count in range(10):
    flip_edge = np.random.choice(range(num_edges))
    edge_index, flag = edgeflip.flip_edge_by_index(edge_index, flip_edge)
    if flag:
        count_flips += 1
