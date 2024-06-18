#  随机选择、贪婪选择

from torch.distributions.categorical import Categorical


def select_action(p):
    dist = Categorical(p.squeeze())
    s = dist.sample()
    return s

def greedy_select_action(p):
    _, index = p.squeeze().max(1)
    # action = candidate[index]
    return index

def sample_select_action(p):
    dist = Categorical(p.squeeze())
    s = dist.sample()
    return s

