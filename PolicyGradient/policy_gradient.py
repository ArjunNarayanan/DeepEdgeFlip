import torch
from torch.nn import ReLU
from torch.nn.functional import softmax, log_softmax
import edgeflip
import game_env
from torch_geometric.nn import GCNConv, Sequential
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from torch_geometric.data import Data


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
        x, edge_index = env.x, env.edge_index
        x = self.gcn(x, edge_index)

        src, dst = edge_index
        logits = (x[src] * x[dst]).sum(dim=-1)

        return logits


def policy_gradient_loss(logits, actions, weights):
    logp = log_softmax(logits, dim=1).gather(1, actions.view(-1, 1))
    return -(logp * weights).mean()


def collect_batch_trajectories(env, policy, batch_size):
    batch_logits = []
    batch_weights = []
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


arch = [1, 4, 4]

num_epochs = 10
batch_size = 5
learning_rate = 1.0

env = game_env.load_template_env()
policy = Policy(arch)

return_trajectory = run_training_loop(env, policy, batch_size, num_epochs, learning_rate)

# optimizer = Adam(policy.parameters(), lr=learning_rate)
#
# batch_logits, batch_actions, batch_weights, average_return = collect_batch_trajectories(env, policy, batch_size)
# loss = policy_gradient_loss(batch_logits, batch_actions, batch_weights)
# loss.backward()
# optimizer.step()
#
# env.reset()
# logits = policy(env)
# probs = softmax(logits, dim=0)

# lr = 0.01
# batch_size=1
# optimizer = Adam(policy.parameters(), lr=lr)
#
# return_trajectory = training_loop(env, policy, optimizer, numepochs, batch_size)
#
# import matplotlib.pyplot as plt
# plt.plot(return_trajectory)

# env.reset()
# batch_probs, batch_acts, batch_weights, average_return = collect_batch_trajectories(env, policy, batch_size)
# loss = [policy_gradient_loss(batch_probs[i], batch_acts[i], batch_weights[i]) for i in range(len(batch_probs))]
#
# mean_loss = torch.mean(torch.tensor(loss, requires_grad=True))
# mean_loss.backward()
# optimizer.step()


# probs = Categorical(logits=policy(env)).probs
# print(probs)
