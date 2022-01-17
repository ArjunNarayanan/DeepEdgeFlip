import torch
from torch.nn import ReLU
import game_env
from torch_geometric.nn import GCNConv, Sequential
from torch.distributions.categorical import Categorical
from copy import deepcopy
from torch.optim import Adam


def make_network(arch):
    gcn = []
    for idx in range(len(arch) - 2):
        gcn.append((GCNConv(arch[idx], arch[idx + 1]), "x, edge_index -> x"))
        gcn.append(ReLU())

    gcn.append((GCNConv(arch[-2], arch[-1]), "x, edge_index -> x"))

    model = Sequential("x, edge_index", gcn)
    return model


class Policy(torch.nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.gcn = make_network(arch)

    def forward(self, env):
        x, edge_index = env.vertex_score, env.edge_index
        x = self.gcn(x, edge_index)

        src, dst = edge_index
        logits = (x[src] * x[dst]).sum(dim=-1)

        return logits

def policy_gradient_loss(probabilities, action, weights):
    logp = probabilities.log_prob(action)
    return -(logp * weights).mean()


def train_one_epoch(policy, nflips = 10, batch_size = 32, nsteps = 100):
    batch_obs = []
    batch_acts = []
    batch_weights = []
    batch_rets = []
    batch_lens = []
    ep_rewards = []

    obs = game_env.GameEnv(nflips)
    done = False
    counter = 0

    while counter < nsteps:
        batch_obs.append(deepcopy(env))

        logits = policy(env)
        probabilities = Categorical(logits=logits)

        action = probabilities.sample()
        reward, done = env.step(action)

        batch_acts.append(action)
        ep_rewards.append(reward)

        counter += 1

    return batch_acts, ep_rewards



env = game_env.GameEnv(10)

arch = [1, 4]
policy = Policy(arch)

batch_acts, ep_rewards = train_one_epoch(policy)


# logits = policy(env)
# probabilities = Categorical(logits=logits)
#
# action = probabilities.sample()
#
# reward, done = env.step(action)
# loss = policy_gradient_loss(probabilities, action, reward)
# loss.backward()