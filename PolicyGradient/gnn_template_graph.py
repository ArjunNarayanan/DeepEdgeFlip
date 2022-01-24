import torch

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
