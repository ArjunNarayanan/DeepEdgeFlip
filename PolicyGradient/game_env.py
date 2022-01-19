import torch
import load_ideal_graph
import edgeflip
from torch_geometric.utils import degree
from copy import deepcopy


def vertex_degree(edge_index):
    deg = degree(edge_index[0]).reshape([-1, 1])
    return deg


def env_score(vertex_score):
    return vertex_score.abs().sum()


class GameEnv():
    def __init__(self, edge_index, desired_degree, random_flips, max_flips):

        self.initial_edge_index = edgeflip.random_flips(edge_index, random_flips)
        self.max_flips = max_flips
        self.num_flips = 0
        self.desired_degree = desired_degree

        self.edge_index = deepcopy(self.initial_edge_index)
        self.degree = vertex_degree(self.edge_index)
        self.x = self.degree - self.desired_degree
        self.score = env_score(self.x)
        self.num_actions = self.edge_index.shape[1]

    def reset(self):
        self.edge_index = deepcopy(self.initial_edge_index)
        self.degree = vertex_degree(self.edge_index)
        self.x = self.degree - self.desired_degree
        self.score = env_score(self.x)
        self.num_flips = 0

    def step(self, action, no_flip_reward=-10.0, done_reward=0.0):

        if self.num_flips >= self.max_flips:
            return torch.tensor(done_reward), True
        else:
            old_score = self.score
            self.edge_index, success = edgeflip.flip_edge_by_index(self.edge_index, action)

            if success:
                self.degree = vertex_degree(self.edge_index)
                self.x = self.degree - self.desired_degree
                self.score = env_score(self.x)
                reward = old_score - self.score
            else:
                reward = torch.tensor(no_flip_reward)

            self.num_flips += 1
            done = False
            if self.score == 0 or self.num_flips == self.max_flips:
                done = True

            return reward, done


def load_game_env_from_file(foldername, random_flips, max_flips):
    edge_index, desired_degree = load_ideal_graph.load_graph(foldername)
    return GameEnv(edge_index, desired_degree, random_flips, max_flips)


def load_template_env():
    edge_index = torch.tensor([[0,0,0,1,1,2,2,2,3,3],
                               [1,2,3,0,2,0,1,3,0,2]])
    desired_degree = torch.tensor([2,3,2,3]).reshape([-1,1])
    return GameEnv(edge_index,desired_degree,0,1)