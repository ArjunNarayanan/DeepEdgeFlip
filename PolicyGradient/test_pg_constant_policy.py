import torch
from torch.nn import Linear
from torch.nn.functional import softmax, log_softmax
from torch.optim import Adam

class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = Linear(1, 2)

    def forward(self):
        return self.lin(torch.tensor([1.0]))


def reward_function(actions):
    reward = torch.clone(actions).float()
    return reward


def collect_batch_trajectories(policy, batch_size):
    batch_logits = []
    batch_weights = []
    batch_actions = []

    counter = 0
    while True:
        logits = policy()
        probs = softmax(logits, dim=0)
        action = torch.multinomial(probs, 1)

        batch_actions.append(action)
        batch_logits.append(logits.unsqueeze(0))

        reward = reward_function(action)
        batch_weights.append(reward)

        counter += 1

        if counter >= batch_size:
            break

    batch_logits = torch.cat(batch_logits)
    batch_weights = torch.tensor(batch_weights)
    batch_actions = torch.tensor(batch_actions)

    return batch_logits, batch_actions, batch_weights


def policy_gradient_loss(logits, actions, weights):
    logp = log_softmax(logits, dim=1).gather(1, actions.view(-1, 1))
    return -(logp * weights).mean()


def step_epoch(policy, optimizer, batch_size):
    optimizer.zero_grad()

    batch_logits, batch_actions, batch_weights = collect_batch_trajectories(policy, batch_size)
    loss = policy_gradient_loss(batch_logits, batch_actions, batch_weights)

    loss.backward()
    optimizer.step()

    avg_reward = batch_weights.mean()
    return avg_reward


def run_training_loop(policy, batch_size, num_epochs, learning_rate):
    optimizer = Adam(policy.parameters(), lr=learning_rate)
    history = []

    for epoch in range(num_epochs):
        avg_reward = step_epoch(policy,optimizer,batch_size)
        history.append(avg_reward)

    return history

policy = Policy()

learning_rate = 0.1
batch_size = 10
num_epochs = 100

history = run_training_loop(policy,batch_size,num_epochs,learning_rate)

import matplotlib.pyplot as plt
fig,ax = plt.subplots()
ax.plot(history)
ax.set_xlabel("Epochs")
ax.set_ylabel("Average return")
ax.set_title("Average return vs epochs for 2 state, fixed returns")
fig.savefig("results\\constant-policy.png")

probs = softmax(policy(),dim=0)