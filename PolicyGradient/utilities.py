import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_return_history(history, title="", filename="", optimum_return = 0.2469):
    fig, ax = plt.subplots()
    ax.plot(history,label="average return")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Average return")
    ax.set_title(title)
    ax.plot(np.repeat(np.array([optimum_return]),len(history)),"--",color="black",label="optimum")
    ax.legend()
    if len(filename) > 0:
        fig.savefig(filename)
    else:
        return fig


def moving_average(a, n=3):
    ret = torch.cumsum(a, dim=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
