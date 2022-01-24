import torch
from torch.nn import ReLU
from torch_geometric.nn import GCNConv, Sequential
from torch.nn.functional import softmax


def make_network(arch):
    gcn = []
    for idx in range(len(arch) - 2):
        gcn.append((GCNConv(arch[idx], arch[idx + 1]), "x, edge_index -> x"))
        gcn.append(ReLU())

    gcn.append((GCNConv(arch[-2], arch[-1]), "x, edge_index -> x"))

    model = Sequential("x, edge_index", gcn)
    return model


class GNNPolicy(torch.nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.gcn1 = make_network(arch)
        self.gcn2 = make_network(arch)

    def forward(self, env):
        x, edge_index = env.x, env.edge_index

        y1 = self.gcn1(x, edge_index)
        y2 = self.gcn2(x, edge_index)

        src, dst = env.src, env.dst

        l1 = (y1[src] * y1[dst]).sum(dim=0)
        l2 = (y2[src] * y2[dst]).sum(dim=0)

        return torch.cat([l1.unsqueeze(0), l2.unsqueeze(0)])


import policy_gradient
import gnn_template_graph

arch = [1, 4, 4]
num_epochs = 200
batch_size = 32
learning_rate = 0.1

env = gnn_template_graph.GameEnv()
policy = GNNPolicy(arch)


history = policy_gradient.run_training_loop(env, policy, batch_size, num_epochs, learning_rate)

# test_env = make_template_with_score([-1, 1, -1, 1])
# logit = policy(env)
# prob = softmax(policy(test_env),dim=0)
# print(prob)
#
# import utilities
# avg_history = utilities.moving_average(torch.tensor(history), n=5)
# filename = "results\\1-4-gnn-template-return"
# utilities.plot_return_history(avg_history,filename=filename)
