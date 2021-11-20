# learn how to decline std in nn
import numpy as np

from alg_GLOBALS import *
from alg_nets import ActorNet
from mpl_toolkits.mplot3d import axes3d


def get_target(input_values):
    # target = np.sin(np.sum(input_values_np, 1))
    target = 0.5 * (np.sin(input_values[:, 0]) + np.cos(input_values[:, 1])) + np.random.normal(
        loc=0.0, scale=0.5, size=input_values[:, 0].shape
    )
    return target


def get_output(net, input_values):
    # value = net(torch.tensor(input_values))
    # return value.squeeze()
    mean, std = net(torch.tensor(input_values))
    output_dist = torch.distributions.Normal(mean, std)
    return output_dist.rsample().squeeze(), mean.squeeze(), std.squeeze()
    # return mean.squeeze(), mean.squeeze(), std.squeeze()


if __name__ == '__main__':
    actor = ActorNet(obs_size=2, n_actions=1)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=0.0001)

    fig = plt.figure(figsize=plt.figaspect(.5))
    fig.suptitle('Analysis oof STD')
    ax_1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax_2 = fig.add_subplot(1, 3, 2)
    ax_3 = fig.add_subplot(1, 3, 3)

    mean_list, std_list, loss_list = [], [], []
    # ax_1.scatter(input_values_np[:, 0], input_values_np[:, 1], output_values_mean, marker='.')
    # ax_1.scatter(input_values_np[:, 0], input_values_np[:, 1], target_values, marker='.')
    # plt.show()

    for i in range(4000):
        input_values_np = np.random.uniform(-5, 5, size=(32, 2))
        z_target = get_target(input_values_np)
        z_target_tensor = torch.tensor(z_target).float()

        z_output_tensor, actor_mean, actor_std = get_output(actor, input_values_np)
        # actor_output_tensor = get_output(actor, input_values_np)

        loss = nn.MSELoss()(z_output_tensor, z_target_tensor)
        actor_optim.zero_grad()
        loss.backward()
        actor_optim.step()

        # PLOT
        # mean_list.append(actor_output_tensor.mean().detach().squeeze().item())
        mean_list.append(actor_mean.mean().detach().squeeze().item())
        std_list.append(actor_std.mean().detach().squeeze().item())
        loss_list.append(loss.item())

        if i % 50 == 0:
            # AX 1
            ax_1.cla()
            input_values_np = np.random.uniform(-5, 5, size=(1800, 2))
            x = input_values_np[:, 0]
            y = input_values_np[:, 1]
            z_target = get_target(input_values_np)

            z_output_tensor, _, _ = get_output(actor, input_values_np)
            # actor_output_tensor = get_output(actor, input_values_np)

            z_output = z_output_tensor.detach().squeeze().numpy()
            ax_1.scatter(x, y, z_target, marker='.', alpha=0.1)
            ax_1.scatter(x, y, z_output, marker='.')

            # AX 2
            ax_2.cla()
            ax_2.plot(mean_list, label='mean')
            ax_2.plot(std_list, label='std')
            ax_2.legend()

            # AX 3
            ax_3.cla()
            ax_3.plot(loss_list)
            ax_3.set_title('Loss')

            plt.pause(0.05)

    # SAVE
    path_to_save = f'data/research_actor_with_std.pt'
    print(f"Saving research_actor's model...")
    torch.save(actor, path_to_save)

    input()
    plt.close()






