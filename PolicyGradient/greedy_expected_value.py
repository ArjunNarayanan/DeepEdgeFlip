import numpy as np

template = np.array([-1,0,1])
col1 = np.tile(template,27)

col2 = np.tile(np.repeat(template,3),9)
col3 = np.tile(np.repeat(template,9),3)
col4 = np.repeat(template,27)

states = np.vstack([col1,col2,col3,col4]).transpose()

old_scores = np.sum(np.abs(states),1)

new_states = np.copy(states)
new_states[:,[0,2]] -= 1
new_states[:,[1,3]] += 1

new_scores = np.sum(np.abs(new_states),1)

score_change = old_scores - new_scores
expected_return = score_change[score_change > 0].sum()/len(score_change)