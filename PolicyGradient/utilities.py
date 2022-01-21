import matplotlib.pyplot as plt
import torch


def plot_return_history(history, title="", filename=""):
    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Average return")
    ax.set_title(title)
    if len(filename) > 0:
        fig.savefig(filename)
    else:
        return fig


def moving_average(a, n=3):
    ret = torch.cumsum(a, dim=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
