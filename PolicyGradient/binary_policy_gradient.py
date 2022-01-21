import torch
from torch.optim import Adam


def policy_gradient_loss(probs, actions, weights):
    logp = (probs.log()).gather(1, actions.view(-1, 1))
    return -(logp * weights).mean()


def collect_batch_trajectories(env, policy, batch_size):
    batch_logits = []
    batch_actions = []
    batch_weights = []
    batch_rets = []
    batch_lens = []
    ep_rewards = []

    while True:
        logit = policy(env)
        prob = logit.sigmoid()
        action = 1 if prob > 0.5 else 0

        reward, done = env.step(action)

        batch_logits.append(logit.unsqueeze(0))
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
                ep_rewards = []

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
