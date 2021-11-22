import numpy as np
import torch

from alg_GLOBALS import *
from alg_nets import ActorNet
from mpl_toolkits.mplot3d import axes3d


def main():
    mean, std = actor(state_tensor)
    actor_dist = torch.distributions.Normal(mean, std)
    prob_1 = torch.exp(actor_dist.log_prob(torch.tensor(1)))
    prob_2 = torch.exp(actor_dist.log_prob(torch.tensor(10)))

    print(f'mean: {mean.item()}, std: {std.item()}, p 1: {prob_1.item()}, p_2: {prob_2.item()}')


if __name__ == '__main__':
    torch.manual_seed(10)
    actor = ActorNet(2, 1)
    env = gym.make(ENV_NAME)
    state = env.reset()
    state_tensor = torch.tensor(state).float()
    main()