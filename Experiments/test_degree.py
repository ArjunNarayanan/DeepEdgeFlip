import torch
from torch_geometric.utils import to_undirected
from torch_geometric.utils import degree


num_nodes = 4
edge_index = torch.tensor([[0, 0, 0, 1, 2],
                           [1, 2, 3, 2, 3]], dtype=torch.long)
edge_index = to_undirected(edge_index)

deg = degree(edge_index[0])