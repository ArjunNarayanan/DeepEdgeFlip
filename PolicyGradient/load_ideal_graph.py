import torch
import pandas as pd
from torch_geometric.utils import to_undirected


def load_graph(foldername):
    connectivity_file = foldername + "connectivity.csv"
    degree_file = foldername + "degree.csv"

    connectivity = pd.read_csv(connectivity_file, header=None)
    edge_tensor = torch.tensor(connectivity.values, dtype=torch.long)
    edge_index = to_undirected(edge_tensor.t().contiguous())

    degree = pd.read_csv(degree_file, header=None)
    degree = torch.tensor(degree.values, dtype=torch.float)

    return edge_index, degree

