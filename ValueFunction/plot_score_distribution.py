import pandas as pd

def load_reward_history(mesh_num, max_reward_per_step=4):
    rewardfile = "reward-history\\mesh-" + str(mesh_num) + ".csv"
    reward_hist = pd.read_csv(rewardfile, header=None)
    numsteps = reward_hist.shape[0]
    r = (reward_hist.sum().values[0]) / (max_reward_per_step * numsteps)
    return r

max_mesh_num = 2900
scores = [load_reward_history(i) for i in range(1,max_mesh_num)]

import matplotlib.pyplot as plt
fig,ax = plt.subplots()
ax.hist(scores,density=True)
ax.grid()
ax.set_xlabel("Normalized return")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of returns for meshes in the dataset")
fig.tight_layout()
fig.savefig("results\\mesh-returns-distribution.png")