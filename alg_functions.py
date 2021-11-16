import numpy as np

from GLOBALS import *


def moving_average(x, w):
    w = int(w)
    if w > 0:
        return np.convolve(x, np.ones(w), 'valid') / w
    return [0]

"""
# H(P, Q) = H(P) + KL(P || Q)
# Where H(P, Q) is the cross-entropy of Q from P, 
# H(P) is the entropy of P and 
# KL(P || Q) is the divergence of Q from P.
"""


# calculate entropy H(P)
def entropy(p):
    return -sum([p[i] * log(p[i]) for i in range(len(p))])


# calculate cross entropy
def cross_entropy(p, q):
    return -sum(pp * log(qq) for pp, qq in zip(p, q))


# calculate the kl divergence KL(P || Q)
def kl_divergence(p, q):
    if len(p.shape) and len(q.shape):
        return sum(p[i] * log(p[i]/q[i]) for i in range(len(p)))
    return 0


def soft_update(target, source, tau, plotter=None):
    # for target_param, param in zip(target_critic.parameters(), critic.parameters()):
    #     target_param.data.copy_(POLYAK * target_param.data + (1.0 - POLYAK) * param.data)
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    if plotter:
        # plotter.plots_update_entropy(source, target, 'critic')
        # plotter.neptune_plot({'entropy': critic_loss.item(), 'loss_actor': actor_loss.item()})
        pass


def OUNoise():
    theta = 0.15
    sigma = 0.3
    mu = 0
    state = 0
    while True:
        yield state
        state += theta * (mu - state) + sigma * np.random.randn()


def matrix_get(net):
    big_list = np.zeros(1)
    for layer in list(net.parameters()):
        layer_np = layer.data.numpy().copy()
        if len(layer_np.shape) > 1:
            layer_np = layer_np.reshape(np.dot(*layer_np.shape))
        else:
            layer_np = layer_np.reshape(layer_np.shape[0])
        big_list = np.concatenate((big_list, layer_np), axis=0)

    to_add = np.zeros(64 - (big_list.shape[0] % 64))
    big_list = np.concatenate((big_list, to_add), axis=0)
    times64 = int(big_list.shape[0] / 64)
    matrix = []
    for t in range(times64):
        one_part = big_list[t*64:(t+1)*64]
        matrix.append(one_part)
    matrix = np.array(matrix)
    return matrix.copy()


def matrix_square_sum(mat):
    mat1 = np.square(mat)
    mat2 = np.sum(mat1)
    return mat2


def matrix_get_diff(net1, net2):
    mat1 = matrix_get(net1)
    mat2 = matrix_get(net2)
    mat3 = mat1 - mat2
    return mat3


def matrix_mse_nets(net1, net2):
    mat_diff = matrix_get_diff(net1, net2)
    output = matrix_square_sum(mat_diff)
    return output


def matrix_mse_mats(mat1, mat2):
    mat_diff = mat1 - mat2
    mat_output = matrix_square_sum(mat_diff)
    return mat_output


def get_cube_of_weights(net, layer_indx):
    layer = list(net.parameters())[layer_indx].data.numpy().copy()
    line_num = np.dot(*layer.shape)
    side = int(np.sqrt(line_num))
    _layer_row = layer.reshape(line_num)[:side ** 2]
    return _layer_row.reshape((side, side)), _layer_row
