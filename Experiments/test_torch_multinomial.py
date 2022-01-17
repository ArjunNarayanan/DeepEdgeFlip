import torch

probability = torch.tensor([0.1,0.1,0.5,0.3])
idx = torch.multinomial(probability,100000,replacement=True)

emp_probability = [sum(idx == i).item()/len(idx) for i in range(4)]