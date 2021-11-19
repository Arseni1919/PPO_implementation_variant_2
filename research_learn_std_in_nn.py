# learn how to decline std in nn
import numpy as np

from alg_GLOBALS import *
from alg_nets import ActorNet
from mpl_toolkits.mplot3d import axes3d

if __name__ == '__main__':
    actor = ActorNet(obs_size=2, n_actions=1)
    input_values_np = np.random.uniform(0, 10, size=(1800, 2))
    target_values = np.sin(np.sum(input_values_np, 1))
    input_values_tensor = torch.tensor(input_values_np)
    output_values_mean, output_values_mean_std = actor(input_values_tensor)
    output_values_mean = output_values_mean.detach().squeeze().numpy()
    print(output_values_mean)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(input_values_np[:, 0], input_values_np[:, 1], output_values_mean, marker='.')
    # ax.scatter(input_values_np[:, 0], input_values_np[:, 1], target_values, marker='.')
    # plt.show()

    for i in range(100):
        ax.cla()
        # x, y, z = axes3d.get_test_data(0.05)
        input_values_np = np.random.uniform(-100, 100, size=(5800, 2))
        target_values = np.sin(np.sum(input_values_np, 1))

        x = input_values_np[:, 0]
        y = input_values_np[:, 1]
        z_target = target_values

        output_values_mean, output_values_mean_std = actor(torch.tensor(input_values_np))
        z_output = output_values_mean.detach().squeeze().numpy()
        # ax.scatter(x, y, z_target, marker='.')
        ax.scatter(x, y, z_output, marker='.')

        plt.pause(1)






