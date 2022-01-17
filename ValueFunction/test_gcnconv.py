import torch
from torch.nn import ReLU
from torch_geometric.nn import GCNConv, Sequential, global_add_pool
import torch.optim as optim
import load_graphs
import numpy as np


def make_network(arch):
    gcn = []
    for idx in range(len(arch) - 2):
        gcn.append((GCNConv
                    (arch[idx], arch[idx + 1]), "x, edge_index -> x"))
        gcn.append(ReLU())

    gcn.append((GCNConv(arch[-2], arch[-1]), "x, edge_index -> x"))

    model = Sequential("x, edge_index", gcn)
    return model


class GCN(torch.nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.gcn = make_network(arch)

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        x = self.gcn(x, edge_index)

        return global_add_pool(x, batch.batch)


def make_network(arch):
    gcn = []
    for idx in range(len(arch) - 2):
        gcn.append((GCNConv(arch[idx], arch[idx + 1]), "x, edge_index -> x"))
        gcn.append(ReLU())

    gcn.append((GCNConv(arch[-2], arch[-1]), "x, edge_index -> x"))
    model = Sequential("x, edge_index", gcn)
    return model


def train_model(model, train_set, test_set, lr=0.001, num_epochs=50):
    loss_function = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss_history = np.zeros([num_epochs, 1])
    test_loss_history = np.zeros([num_epochs, 1])
    test_accuracy = np.zeros([num_epochs, 1])

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for idx, batch in enumerate(train_set):
            optimizer.zero_grad()

            prediction = model(batch)
            loss = loss_function(prediction, batch.y)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss_history[epoch] = train_loss / len(train_set)

        model.eval()

        with torch.no_grad():
            batch = next(iter(test_set))
            test_pred = model(batch)
            test_loss_history[epoch] = loss_function(test_pred, batch.y).item()

        test_pred = test_pred.numpy()
        ground_truth = batch.y.numpy()
        accuracy = measure_accuracy(test_pred, ground_truth)
        test_accuracy[epoch] = accuracy

        print("epoch = %d \t train loss = %.4f \t test loss = %.4f \t accuracy = %.1f" % (
            epoch, train_loss_history[epoch], test_loss_history[epoch], accuracy))

    return train_loss_history, test_loss_history, test_accuracy


def measure_accuracy(prediction, truth):
    n = len(prediction)
    assert len(truth) == n

    flags = np.zeros([n, n], dtype=bool)
    for i in range(n):
        for j in range(i, n):
            pi = prediction[i, 0]
            pj = prediction[j, 0]

            ti = truth[i, 0]
            tj = truth[j, 0]

            if (pi < pj and ti < tj) or (pi > pj and ti > tj) or (pi == pj and ti == tj):
                flags[i, j] = True

    num_correct = np.count_nonzero(flags)
    total_comparisons = n * (n + 1) / 2
    accuracy = num_correct / total_comparisons * 100
    return accuracy


num_meshes = 2000
train_ratio = 0.8

train_loader, test_loader = load_graphs.load_all_graphs(num_meshes, batch_size=32, train_ratio=train_ratio)

arch = [1, 4, 4, 1]
model = GCN(arch)
lr = 0.001
train_loss, test_loss, test_accuracy = train_model(model, train_loader, test_loader, lr=lr, num_epochs=50)

# import matplotlib.pyplot as plt
#
# architecture = "1-4-4-4-1"
#
# fig,ax = plt.subplots()
# ax.plot(train_loss,label="Train Loss")
# ax.set_yscale("log")
# ax.plot(test_loss,label="Test Loss")
# ax.set_xlabel("Epochs")
# ax.set_ylabel("Loss")
# ax.set_title(architecture + " architecture")
# ax.legend()
# fig.savefig("train-test-loss-" + architecture + ".png")
#
# fig,ax = plt.subplots()
# ax.plot(test_accuracy)
# ax.set_xlabel("Epochs")
# ax.set_ylabel("Test accuracy")
# ax.set_ylim([0,100])
# ax.set_title(architecture + " architecture")
# fig.savefig("test-accuracy-" + architecture + ".png")
