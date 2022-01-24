import torch
from torch_geometric.nn import GATConv
from torch.nn import Linear
import gnn_template_graph


class GATPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gat1 = GATConv(1, 4, 2)
        self.gat2 = GATConv(8, 4, 2)
        self.linear = Linear(16, 2)

    def forward(self, env):
        x, edge_index = env.x, env.edge_index

        x = self.gat1(x, edge_index)
        x = self.gat2(x, edge_index)

        src, dst = env.src, env.dst

        y = torch.cat([x[src], x[dst]])
        y = self.linear(y)

        return y


import policy_gradient

policy = GATPolicy()
env = gnn_template_graph.GameEnv()

y = policy(env)

batch_size = 32
num_epochs = 100
learning_rate = 0.1

history = policy_gradient.run_training_loop(env, policy, batch_size, num_epochs, learning_rate)

# import utilities
# utilities.plot_return_history(torch.tensor(history))

probs = torch.nn.functional.softmax(policy(env), dim=0)
