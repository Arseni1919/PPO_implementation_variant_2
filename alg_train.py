import torch

from alg_plotter import ALGPlotter
from alg_env_wrapper import SingleAgentEnv
from alg_nets import *
from alg_replay_buffer import ReplayBuffer
from alg_play import load_and_play, play, get_action
from alg_functions import *


def get_train_action(net, observation):
    action_mean, action_std = net(observation)
    action_dist = torch.distributions.Normal(action_mean, action_std)
    action = action_dist.sample()
    clipped_action = torch.clamp(action, min=-1, max=1)
    return clipped_action


def train():
    plotter.info('Training...')

    # --------------------------- # MAIN LOOP # -------------------------- #
    for i_update in range(N_UPDATES):
        plotter.info(f'Update {i_update + 1}')
        minibatch = []
        p = i_update / N_UPDATES

        # COLLECT SET OF TRAJECTORIES
        for i_episode in range(N_EPISODES_PER_UPDATE):
            trajectory, episode_score = get_trajectory(p)
            minibatch.append(trajectory)
            print(f'\r(episode {i_episode + 1}), episode score: {episode_score}', end='' if i_episode != N_EPISODES_PER_UPDATE - 1 else '\n')

        # COMPUTE REWARDS-TO-GO
        rewards_to_go, critic_values, advantages, observations, actions = [], [], [], [], []
        for traj in minibatch:
            i_observations, i_actions, i_rewards, i_dones, i_new_observations = zip(*traj)
            i_rewards_to_go = compute_rewards_to_go(i_rewards)
            i_rewards_to_go = torch.stack(i_rewards_to_go).squeeze()
            rewards_to_go.append(i_rewards_to_go)
            i_critic_values = [critic(observation).detach() for observation, action in zip(i_observations, i_actions)]
            critic_values.append(i_critic_values)
            i_observations = torch.stack(i_observations).squeeze()
            observations.append(i_observations)
            i_actions = torch.stack(i_actions).squeeze()
            actions.append(i_actions)

            # COMPUTE ADVANTAGES
            i_critic_values = torch.stack(i_critic_values).squeeze()
            # i_critic_values = torch.ones_like(i_rewards_to_go) * i_rewards_to_go.mean().item()
            # i_advantages = i_rewards_to_go
            i_advantages = i_rewards_to_go - i_critic_values
            advantages.append(i_advantages)

        # CONCATENATE ALL
        advantages = torch.cat(advantages).detach()
        observations = torch.cat(observations).detach()
        actions = torch.cat(actions).detach()
        rewards_to_go = torch.cat(rewards_to_go).detach()

        # UPDATE ACTOR
        mean_old, std_old = actor_old(observations)
        action_dist_old = torch.distributions.Normal(mean_old.squeeze(), std_old.squeeze())
        action_log_probs_old = action_dist_old.log_prob(actions).detach()

        mean, std = actor(observations)
        action_dist = torch.distributions.Normal(mean.squeeze(), std.squeeze())
        action_log_probs = action_dist.log_prob(actions)

        ratio_of_probs = torch.exp(action_log_probs - action_log_probs_old)
        surrogate1 = ratio_of_probs * advantages
        surrogate2 = torch.clamp(ratio_of_probs, 1 - EPSILON, 1 + EPSILON) * advantages
        loss_actor = torch.min(surrogate1, surrogate2)

        # ADD ENTROPY TERM
        actor_dist_entropy = action_dist.entropy().detach()
        loss_actor = loss_actor - actor_dist_entropy

        loss_actor = - loss_actor.mean()
        actor_optim.zero_grad()
        loss_actor.backward()
        # actor_list_of_grad = [torch.max(torch.abs(param.grad)).item() for param in actor.parameters()]
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 40)
        actor_optim.step()

        # UPDATE CRITIC
        critic_values = critic(observations).squeeze()
        loss_critic = nn.MSELoss()(critic_values, rewards_to_go)
        critic_optim.zero_grad()
        loss_critic.backward()
        critic_optim.step()

        # UPDATE OLD NET
        if i_update % 5 == 0:
            actor_old.load_state_dict(actor.state_dict())

        # PLOTTER
        # plotter.neptune_plot({'actor_dist_entropy_mean': actor_dist_entropy.mean().item()})
        # plotter.neptune_plot({'actor_mean': mean.mean().item(), 'actor_std': std.mean().item()})
        # plotter.neptune_plot({'loss_actor': loss_actor.item()})
        # plotter.neptune_plot({'loss_critic': loss_critic.item()})
        # plotter.neptune_plot({'actor_max_grad': max(actor_list_of_grad)})
        # mat1 = plotter.matrix_get_prev('actor')
        # mat2 = matrix_get(actor)
        # mse_actor = matrix_mse_mats(mat1, mat2)
        # plotter.neptune_plot({'mse_actor': mse_actor})

        plotter.matrix_update('critic', critic)
        plotter.matrix_update('actor', actor)

        # RENDER
        if i_update % 4 == 0 and i_update > 0:
            # play(env, 1, actor)
            pass
        # mean, std, loss_actor = [], [], []
        plot_graphs(mean, std, loss_actor, loss_critic, i_update, actions, observations, critic_values)

    # ---------------------------------------------------------------- #

    # FINISH TRAINING
    plotter.close()
    plt.close()
    env.close()
    plotter.info('Finished train.')


def plot_graphs(actor_mean, actor_std, loss, loss_critic, i, actor_output_tensor, input_values_tensor, critic_output_tensor):
    # PLOT
    mean_list.append(actor_output_tensor.mean().detach().squeeze().item())
    mean_list.append(actor_mean.mean().detach().squeeze().item())
    std_list.append(actor_std.mean().detach().squeeze().item())
    loss_list_actor.append(loss.item())
    loss_list_critic.append(loss_critic.item())

    if i % 2 == 0:
        # AX 1
        ax_1.cla()
        input_values_np = input_values_tensor.squeeze().numpy()
        x = input_values_np[:, 0]
        y = input_values_np[:, 1]

        # actor_output_tensor_np = actor_output_tensor.detach().squeeze().numpy()
        # ax_1.scatter(x, y, actor_output_tensor_np, marker='.', label='actions')
        critic_output_tensor_np = critic_output_tensor.detach().squeeze().numpy()
        ax_1.scatter(x, y, critic_output_tensor_np, marker='.', alpha=0.1, label='critic values')
        ax_1.legend()

        # AX 2
        ax_2.cla()
        ax_2.plot(mean_list, label='mean')
        ax_2.plot(std_list, label='std')
        ax_2.legend()

        # AX 3
        ax_3.cla()
        ax_3.plot(loss_list_actor, label='actor')
        ax_3.plot(loss_list_critic, label='critic')
        ax_3.set_title('Loss')
        ax_3.legend()

        plt.pause(0.05)


def compute_rewards_to_go(rewards):
    Val = 0
    Vals = [0] * len(rewards)
    for t in reversed(range(len(rewards))):
        Val = rewards[t] + GAMMA * Val
        Vals[t] = Val
    return Vals


def get_trajectory(p):
    trajectory = []

    # FIRST OBSERVATION
    observation = env.reset()
    done = False
    episode_score = 0

    while not done:
        action = get_train_action(actor_old, observation)
        plotter.neptune_plot({"action": action.item()})
        new_observation, reward, done, info = env.step(action)
        trajectory.append((observation, action, reward, done, new_observation))
        observation = new_observation
        episode_score += reward.item()

    plotter.neptune_plot({"episode_score": episode_score})

    return trajectory, episode_score


def save_results(model_to_save, name):
    path_to_save = f'{SAVE_PATH}/{name}.pt'
    # SAVE
    if SAVE_RESULTS:
        # SAVING...
        plotter.info(f"Saving {name}'s model...")
        torch.save(model_to_save, path_to_save)
    return path_to_save


if __name__ == '__main__':
    # --------------------------- # PLOTTER & ENV # -------------------------- #
    plotter = ALGPlotter(plot_life=PLOT_LIVE, plot_neptune=NEPTUNE, name='my_run_ppo',
                         tags=["PPO", "clip_grad_norm", "b(s) = 0", ENV_NAME])
    env = SingleAgentEnv(env_name=ENV_NAME, plotter=plotter)

    # --------------------------- # NETS # -------------------------- #
    critic = CriticNet(obs_size=env.observation_size(), n_actions=env.action_size())
    actor = ActorNet(obs_size=env.observation_size(), n_actions=env.action_size())
    actor_old = ActorNet(obs_size=env.observation_size(), n_actions=env.action_size())
    actor_old.load_state_dict(actor.state_dict())

    # --------------------------- # OPTIMIZERS # -------------------------- #
    critic_optim = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)

    # --------------------------- # REPLAY BUFFER # -------------------------- #
    # replay_buffer = ReplayBuffer()

    # --------------------------- # NOISE # -------------------------- #
    current_sigma = SIGMA
    normal_distribution = Normal(torch.tensor(0.0), torch.tensor(current_sigma))

    # Simple Ornstein-Uhlenbeck Noise generator
    ou_noise = OUNoise()

    # --------------------------- # PLOTTER INIT # -------------------------- #
    plotter.neptune_set_parameters()
    plotter.matrix_update('critic', critic)
    plotter.matrix_update('actor', actor)
    fig = plt.figure(figsize=plt.figaspect(.5))
    fig.suptitle('MountainCar')
    ax_1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax_2 = fig.add_subplot(1, 3, 2)
    ax_3 = fig.add_subplot(1, 3, 3)

    mean_list, std_list, loss_list_actor, loss_list_critic = [], [], [], []

    # ---------------------------------------------------------------- #
    # ---------------------------------------------------------------- #

    # Main Process
    train()

    # Save
    path_actor_model = save_results(actor, name='actor')

    # Example Plays
    plotter.info('Example run...')
    load_and_play(env, 3, path_actor_model)

    # ---------------------------------------------------------------- #
