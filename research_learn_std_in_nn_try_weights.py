from alg_GLOBALS import *
from alg_nets import ActorNet
from mpl_toolkits.mplot3d import axes3d
from research_learn_std_in_nn import get_target

if __name__ == '__main__':
    # torch.save(actor, f'{SAVE_PATH}/actor.pt')
    # torch.save(target_actor, f'{SAVE_PATH}/target_actor.pt')
    actor_model = torch.load(f'data/research_actor.pt')
    actor_model_2 = torch.load(f'data/research_actor_2.pt')
    actor_model_3 = torch.load(f'data/research_actor_3.pt')

    fig = plt.figure()
    fig.suptitle('Analysis oof STD')
    ax_1 = fig.add_subplot(1, 1, 1, projection='3d')
    input_values_np = np.random.uniform(-5, 5, size=(2800, 2))
    x = input_values_np[:, 0]
    y = input_values_np[:, 1]
    z_target = get_target(input_values_np)

    ax_1.scatter(x, y, z_target, marker='.', alpha=0.1, label='Target')
    ax_1.scatter(x, y, actor_model(torch.tensor(input_values_np)).detach().squeeze().numpy(), marker='.', label='Small NN')
    ax_1.scatter(x, y, actor_model_2(torch.tensor(input_values_np)).detach().squeeze().numpy(), marker='.', label='Middle NN')
    ax_1.scatter(x, y, actor_model_3(torch.tensor(input_values_np)).detach().squeeze().numpy(), marker='.', label='Large NN')
    ax_1.legend()
    plt.show()
