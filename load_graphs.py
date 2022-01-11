import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def load_graph(mesh_num):
    edgefile = "edge-connectivity\\mesh-" + str(mesh_num) + ".csv"
    edges = pd.read_csv(edgefile, header=None)

    vertexfile = "vertex-scores\\mesh-" + str(mesh_num) + ".csv"
    vertexscores = pd.read_csv(vertexfile, header=None)

    edge_tensor = torch.tensor(edges.values, dtype=torch.long)
    vertex_tensor = torch.tensor(vertexscores.values, dtype=torch.float)

    reward = torch.tensor([[load_reward_history(mesh_num)]], dtype=torch.float)

    graph = Data(x=vertex_tensor, edge_index=edge_tensor.t().contiguous(), y=reward)
    return graph


def load_reward_history(mesh_num, max_reward_per_step=4):
    rewardfile = "reward-history\\mesh-" + str(mesh_num) + ".csv"
    reward_hist = pd.read_csv(rewardfile, header=None)
    numsteps = reward_hist.shape[0]
    r = (reward_hist.sum().values[0]) / (max_reward_per_step * numsteps)
    return r


def load_all_graphs(num_meshes, batch_size=32, train_ratio=0.6, test_ratio=0.2):
    mesh_numbers = range(1, num_meshes + 1)
    train_size = int(train_ratio * num_meshes)
    test_size = int()
    test_size = num_meshes - train_size

    all_graphs = [load_graph(i) for i in mesh_numbers]
    train_graphs, test_graphs = torch.utils.data.random_split(all_graphs, [train_size, test_size])

    train_loader = DataLoader(train_graphs, batch_size=batch_size)
    test_loader = DataLoader(test_graphs, batch_size=test_size)
    return train_loader, test_loader
