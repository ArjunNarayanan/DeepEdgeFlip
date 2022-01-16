import torch
from torch_sparse import SparseTensor

edge_index = torch.tensor([[0, 0, 0, 1, 2],
                           [1, 2, 3, 2, 3]], dtype=torch.long)
adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))

mask = torch.tensor([True, False, True, True, True])

adj2 = adj.index_select_nnz(mask)