import torch
from torch.nn import ReLU
from torch.nn.functional import  softmax
import game_env, edgeflip
from torch_geometric.nn import GCNConv, Sequential
from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

def make_network(arch):
    gcn = []
    for idx in range(len(arch) - 2):
        gcn.append((GCNConv(arch[idx], arch[idx + 1]), "x, edge_index -> x"))
        gcn.append(ReLU())

    gcn.append((GCNConv(arch[-2], arch[-1]), "x, edge_index -> x"))

    model = Sequential("x, edge_index", gcn)
    return model



env = game_env.GameEnv(0)


# num_edges = edge_index.shape[1]
#
# edge_index = edgeflip.random_flips(edge_index,30)
# deg = vertex_degree(edge_index)
#
# X1 = deg - ideal_degree
# graph_score1 = sum(abs(X1)).item()
#
#
# arch = [1,4]
# model = make_network(arch)
#
# out = model(X1, edge_index)
# src, dst = edge_index
#
# score = (out[src] * out[dst]).sum(dim=-1)
# probability = softmax(score,dim=0)
#
# idx = torch.multinomial(probability,1).item()
#
# edge_index, success = edgeflip.flip_edge_by_index(edge_index,idx)
# deg = vertex_degree(edge_index)
#
# X2 = deg - ideal_degree
# graph_score2 = sum(abs(X2)).item()