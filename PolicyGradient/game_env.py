import load_ideal_graph
import edgeflip
from torch_geometric.utils import degree

def vertex_degree(edge_index):
    deg = degree(edge_index[0]).reshape([-1,1])
    return deg

def env_score(vertex_score):
    return vertex_score.abs().sum().item()


class GameEnv():
    def __init__(self, nflips, foldername="ideal-mesh\\"):
        edge_index, desired_degree = load_ideal_graph.load_graph(foldername)

        self.edge_index = edgeflip.random_flips(edge_index, nflips)
        self.desired_degree = desired_degree
        self.degree = vertex_degree(self.edge_index)
        self.vertex_score = self.degree - self.desired_degree
        self.score = env_score(self.vertex_score)
        self.num_actions = self.edge_index.shape[1]

    def step(self, action):

        old_score = self.score
        self.edge_index = edgeflip.flip_edge_by_index(self.edge_index, action)
        self.degree = vertex_degree(self.edge_index)
        self.vertex_score = self.degree - self.desired_degree
        self.score = env_score(self.vertex_score)

        reward = old_score - self.score
        done = True if env.score == 0 else False

        return reward, done


