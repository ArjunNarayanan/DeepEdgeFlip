import torch
from torch.nn import ReLU, Linear, Sequential
from torch.nn.functional import softmax, log_softmax
# from torch_geometric.nn import GCNConv, Sequential
from torch.optim import Adam


def make_network(arch):
    gcn = []
    for idx in range(len(arch) - 2):
        gcn.append((GCNConv(arch[idx], arch[idx + 1]), "x, edge_index -> x"))
        gcn.append(ReLU())

    gcn.append((GCNConv(arch[-2], arch[-1]), "x, edge_index -> x"))

    model = Sequential("x, edge_index", gcn)
    return model

def make_mlp_network(arch):
    mlp = []
    for idx in range(len(arch) - 2):
        mlp.append(Linear(arch[idx],arch[idx+1]))
        mlp.append(ReLU())

    mlp.append(Linear(arch[-2], arch[-1]))

    model = Sequential(*mlp)
    return model


class GNNPolicy(torch.nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.gcn = make_network(arch)

    def forward(self, env):
        x, edge_index = env.x, env.edge_index
        x = self.gcn(x, edge_index)

        src, dst = edge_index
        logits = (x[src] * x[dst]).sum(dim=-1)

        return logits


class MLPPolicy(torch.nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.mlp = make_mlp_network(arch)

    def forward(self, env):
        logits = self.mlp(env.x)
        return logits


def policy_gradient_loss(logits, actions, weights):
    logp = log_softmax(logits, dim=1).gather(1, actions.view(-1, 1))
    return -(logp * weights).mean()


def collect_batch_trajectories(env, policy, batch_size):
    batch_logits = []
    batch_actions = []
    batch_weights = []
    batch_rets = []
    batch_lens = []
    ep_rewards = []

    while True:
        logits = policy(env)
        probs = softmax(logits, dim=0)
        action = torch.multinomial(probs, 1)

        reward, done = env.step(action)

        batch_logits.append(logits.unsqueeze(0))
        batch_actions.append(action)
        ep_rewards.append(reward)

        if done:
            ep_ret, ep_len = sum(ep_rewards), len(ep_rewards)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)

            batch_weights += [ep_ret] * ep_len

            if len(batch_actions) >= batch_size:
                break
            else:
                env.reset()
                done, ep_rewards = False, []

    batch_logits = torch.cat(batch_logits)
    batch_weights = torch.tensor(batch_weights)
    batch_actions = torch.tensor(batch_actions)

    average_return = sum(batch_rets) / len(batch_rets)
    return batch_logits, batch_actions, batch_weights, average_return


def step_epoch(env, policy, optimizer, batch_size):
    env.reset()
    optimizer.zero_grad()

    batch_logits, batch_actions, batch_weights, average_return = collect_batch_trajectories(env, policy, batch_size)
    loss = policy_gradient_loss(batch_logits, batch_actions, batch_weights)

    loss.backward()
    optimizer.step()

    return loss, average_return


def run_training_loop(env, policy, batch_size, num_epochs, learning_rate):
    optimizer = Adam(policy.parameters(), lr=learning_rate)
    return_trajectory = []

    for epoch in range(num_epochs):
        loss, average_return = step_epoch(env, policy, optimizer, batch_size)
        print("epoch = %3d \t loss: %.3f \t average return = %3.3f " % (epoch, loss, average_return))
        return_trajectory.append(average_return)

    return return_trajectory


import template_graph

arch = [4, 2]

num_epochs = 100
batch_size = 5
learning_rate = 0.1

env = template_graph.GameEnv()
policy = MLPPolicy(arch)

logits = policy(env)

# history = run_training_loop(env,policy,batch_size,num_epochs,learning_rate)
#
#
# import matplotlib.pyplot as plt
# plt.plot(history)
