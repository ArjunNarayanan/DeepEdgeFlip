import torch

def env_score(x):
    return x.abs().sum()

class GameEnv():
    def __init__(self):
        self.edge_index = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2, 3, 3],
                                        [1, 2, 3, 0, 2, 0, 1, 3, 0, 2]], dtype=torch.long)
        self.x = torch.randint(-1,2,[4,1]).float()
        self.score = env_score(self.x)

    def reset(self):
        self.x = torch.randint(-1,2,[4,1]).float()
        self.score = env_score(self.x)

    def step(self,action,no_flip_reward=0.0):
        if action == 1 or action == 5:
            old_score = self.score
            delta = torch.tensor([-1., +1., -1., +1.]).view([4, 1])
            y = self.x + delta
            new_score = env_score(y)
            reward = old_score - new_score
            return reward, True
        else:
            return torch.tensor(no_flip_reward), True

