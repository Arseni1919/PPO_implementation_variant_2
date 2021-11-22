# ------------------------------------------- #
# ------------------IMPORTS:----------------- #
# ------------------------------------------- #
import os
import time
import logging
from collections import namedtuple, deque
from termcolor import colored
import random
from math import log
import gym
import pettingzoo
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotly
import plotly.express as px
import neptune.new as neptune
from neptune.new.types import File
from dotenv import load_dotenv

import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
# from torchvision import transforms
from torch.distributions import Normal
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset
load_dotenv()

# ------------------------------------------- #
# ------------------FOR ENV:----------------- #
# ------------------------------------------- #
ENV_NAME = "MountainCarContinuous-v0"
# ENV_NAME = "CartPole-v1"
# ENV_NAME = 'LunarLanderContinuous-v2'
# ENV_NAME = "BipedalWalker-v3"
from pettingzoo.mpe import simple_spread_v2
MAX_CYCLES = 25
# MAX_CYCLES = 75
# NUMBER_OF_AGENTS = 3
NUMBER_OF_AGENTS = 1
# ENV = simple_spread_v2.env(N=3, local_ratio=0.5, max_cycles=MAX_CYCLES, continuous_actions=True)
# ENV = simple_spread_v2.parallel_env(N=NUMBER_OF_AGENTS, local_ratio=0.5, max_cycles=MAX_CYCLES, continuous_actions=True)

NUMBER_OF_GAMES = 10
SAVE_RESULTS = True
# SAVE_RESULTS = False
SAVE_PATH = 'data'

# NEPTUNE = True
NEPTUNE = False
# PLOT_LIVE = True
PLOT_LIVE = False
RENDER_WHILE_TRAINING = False

# ------------------------------------------- #
# ------------------FOR ALG:----------------- #
# ------------------------------------------- #

# MAX_LENGTH_OF_A_GAME = 10000
# ENTROPY_BETA = 0.001
# REWARD_STEPS = 4
# CLIP_GRAD = 0.1

# BATCH_SIZE = 64
BATCH_SIZE = 5000
REPLAY_BUFFER_SIZE = BATCH_SIZE * 157
WARMUP = BATCH_SIZE * 3
N_STEPS = REPLAY_BUFFER_SIZE + 10000
# N_EPISODES = 120
N_EPISODES = 70
N_UPDATES = 70
N_EPISODES_PER_UPDATE = 10
LR_CRITIC = 1e-3
LR_ACTOR = 1e-3
GAMMA = 0.995  # discount factor
EPSILON = 0.1
SIGMA = 0.4
LAMBDA = 0.97
TAU = 0.001
VAL_EVERY = 2000
TRAIN_EVERY = 100
HIDDEN_SIZE = 64
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done', 'new_state'])