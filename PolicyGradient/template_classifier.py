import torch
from torch.nn import ReLU
from torch_geometric.nn import GCNConv, Sequential


def make_network(arch):
    gcn = []
    for idx in range(len(arch) - 2):
        gcn.append((GCNConv(arch[idx], arch[idx + 1]), "x, edge_index -> x"))
        gcn.append(ReLU())

    gcn.append((GCNConv(arch[-2], arch[-1]), "x, edge_index -> x"))

    model = Sequential("x, edge_index", gcn)
    return model


class GCNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn = make_network([1, 4, 4])

    def forward(self, env):
        return


