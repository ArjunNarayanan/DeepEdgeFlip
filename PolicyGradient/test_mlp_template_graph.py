import torch
from torch.nn import ReLU, Linear, Sequential


def make_mlp_network(arch):
    mlp = []
    for idx in range(len(arch) - 2):
        mlp.append(Linear(arch[idx], arch[idx + 1]))
        mlp.append(ReLU())

    mlp.append(Linear(arch[-2], arch[-1]))

    model = Sequential(*mlp)
    return model
