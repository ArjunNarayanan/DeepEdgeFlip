import torch
from torch.nn import ReLU, Linear, Sequential
from torch.nn.functional import softmax
import policy_gradient


def make_mlp_network(arch):
    mlp = []
    for idx in range(len(arch) - 2):
        mlp.append(Linear(arch[idx], arch[idx + 1]))
        mlp.append(ReLU())

    mlp.append(Linear(arch[-2], arch[-1]))

    model = Sequential(*mlp)
    return model


class MLPPolicy(torch.nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.mlp = make_mlp_network(arch)

    def forward(self, env):
        logits = self.mlp(env.x)
        return logits


def env_score(x):
    return x.abs().sum()


def get_vertex_scores():
    return torch.randint(-1, 2, [4]).float()


class GameEnv():
    def __init__(self):
        self.x = get_vertex_scores()
        self.score = env_score(self.x)

    def reset(self):
        self.x = get_vertex_scores()
        self.score = env_score(self.x)

    def step(self, action, no_flip_reward=0.0):
        assert action == 1 or action == 0

        if action == 1:
            old_score = self.score
            delta = torch.tensor([-1., +1., -1., +1.])
            y = self.x + delta
            new_score = env_score(y)
            reward = old_score - new_score
            return reward, True
        else:
            return torch.tensor(no_flip_reward), True


def make_template_with_score(vertex_scores):
    assert len(vertex_scores) == 4
    vs = torch.tensor(vertex_scores, dtype=torch.float)

    env = GameEnv()
    env.x = vs
    env.score = env_score(vs)

    return env


arch = [4, 2]
num_epochs = 200
batch_size = 32
learning_rate = 0.1

env = GameEnv()
policy = MLPPolicy(arch)

# history = policy_gradient.run_training_loop(env, policy, batch_size, num_epochs, learning_rate)
#
# import utilities
#
# history = torch.tensor(history)
# avg_history = utilities.moving_average(history, n=5)
# filename = "results\\4-4-2-mlp-template-return.png"
# title = "Average return vs epochs for NN on template graph"
# utilities.plot_return_history(avg_history, title=title, filename=filename)
#
# test_env = make_template_with_score([-1, 1, -1, 1])
# logits = policy(test_env)
# probs = softmax(logits,dim=0)
# print(probs)