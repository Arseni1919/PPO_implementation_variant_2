import os.path
import random

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import torch
from torch.distributions import Normal
from GLOBALS import *
print(type(os.environ['NEPTUNE_API_TOKEN']))
file_path = f'{os.getcwd()}/.neptune'
print(os.getcwd())
if os.path.exists(file_path):
    # removing the file using the os.remove() method
    os.remove(file_path)