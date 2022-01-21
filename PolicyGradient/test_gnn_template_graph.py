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
        op1, op2 = env.op1, env.op2

        l1 = (y1[op1] * y1[op2]).sum(dim=0)
        l2 = (y2[op1] * y2[op2]).sum(dim=0)

        return torch.cat([l1.unsqueeze(0), l2.unsqueeze(0)])


def env_score(x):
    return x.abs().sum()


def get_vertex_scores():
    return torch.randint(-1, 2, [4]).float().reshape(-1, 1)


class GameEnv():
    def __init__(self):
        self.edge_index = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2, 3, 3],
                                        [1, 2, 3, 0, 2, 0, 1, 3, 0, 2]], dtype=torch.long)
        self.x = get_vertex_scores()
        self.score = env_score(self.x)

        self.src = 0
        self.dst = 2
        self.op1 = 1
        self.op2 = 3

    def reset(self):
        self.x = get_vertex_scores()
        self.score = env_score(self.x)

    def step(self, action, no_flip_reward=0.0):
        assert action == 0 or action == 1

        if action == 1:
            old_score = self.score
            delta = torch.tensor([-1., +1., -1., +1.]).reshape(-1,1)
            y = self.x + delta
            new_score = env_score(y)
            reward = old_score - new_score
            return reward, True
        else:
            return torch.tensor(no_flip_reward), True


def make_template_with_score(vertex_scores):
    assert len(vertex_scores) == 4
    vs = torch.tensor(vertex_scores, dtype=torch.float).reshape(-1, 1)

    env = GameEnv()
    env.x = vs
    env.score = env_score(vs)

    return env


import policy_gradient

arch = [1, 4, 4]
num_epochs = 200
batch_size = 32
learning_rate = 0.1

env = GameEnv()
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
