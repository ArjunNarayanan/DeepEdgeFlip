import torch
from torch.nn import ReLU
from torch_geometric.nn import GCNConv, Sequential, global_add_pool
import load_graphs


def make_network(arch):
    gcn = []
    for idx in range(len(arch) - 2):
        gcn.append((GCNConv(arch[idx], arch[idx + 1]), "x, edge_index -> x"))
        gcn.append(ReLU())

    gcn.append((GCNConv(arch[-2], arch[-1]), "x, edge_index -> x"))

    model = Sequential("x, edge_index", gcn)
    return model


class GCN(torch.nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.gcn = make_network(arch)

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        x = self.gcn(x, edge_index)

        return global_add_pool(x, batch.batch)


arch = [1, 4, 8, 4, 1]
model = GCN(arch)

num_meshes = 100
train_ratio = 0.8
train_loader, test_loader = load_graphs.load_all_graphs(num_meshes, batch_size=32, train_ratio=train_ratio)

batch = next(iter(train_loader))
output = model(batch)
