import game_env
import numpy as np

done = False
nsteps = 100

env = game_env.GameEnv(0)

reward_history = []
counter = 0
while not done and counter < nsteps:
    reward, done = env.step(np.random.choice(range(env.num_actions)))
    reward_history.append(reward)
    counter += 1

