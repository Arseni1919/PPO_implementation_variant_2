import numpy as np
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
    # clipped_action = torch.clamp(action, min=-1, max=1)
    return action


def train():
    plotter.info('Training...')
    best_score = -100
    total_scores, total_avg_scores = [0], [0]
    state_stat = running_state(env.reset().detach().squeeze().numpy())
    # --------------------------- # MAIN LOOP # -------------------------- #
    for i_update in range(N_UPDATES):
        plotter.info(f'Update {i_update + 1}')

        with torch.no_grad():
            # SAMPLE TRAJECTORIES
            states, actions, rewards, dones, next_states, average_score = get_trajectories(total_scores, total_avg_scores, state_stat)
            states_tensor = torch.tensor(states).float()
            actions_tensor = torch.tensor(actions).float()
            critic_values_tensor = critic(states_tensor).detach().squeeze()
            critic_values = critic_values_tensor.numpy()

            # COMPUTE RETURNS AND ADVANTAGES
            returns_tensor, advantages_tensor = compute_returns_and_advantages(rewards, dones, critic_values)

        # UPDATE CRITIC
        loss_critic = update_critic(states_tensor, returns_tensor)

        # UPDATE ACTOR
        # mean, std, loss_actor = update_actor(states_tensor, actions_tensor, returns_tensor)
        mean, std, loss_actor = update_actor(states_tensor, actions_tensor, advantages_tensor)


        # PLOTTER
        plot_neptune()
        plot_graphs(
            mean, std, loss_actor, loss_critic, i_update, actions_tensor, states_tensor, critic_values_tensor,
            total_scores, total_avg_scores, state_stat.mean(), state_stat.std()
        )

        # RENDER
        if i_update > N_UPDATES - 5:
            play(env, 1, actor_old)
            # pass

        # SAVE
        if average_score > best_score:
            best_score = average_score
            save_results(path_to_save, actor)

    # ---------------------------------------------------------------- #

    # FINISH TRAINING
    # plotter.close()
    plt.pause(0)
    plt.close()
    env.close()
    plotter.info('Finished train.')


def get_trajectories(scores, scores_avg, state_stat):

    states, actions, rewards, dones, next_states = [], [], [], [], []

    n_episodes = 0
    episode_scores = []

    while not len(rewards) > BATCH_SIZE:
        state = env.reset()

        state_np = state.detach().squeeze().numpy()
        state_stat.update(state_np)
        state_np = np.clip((state_np - state_stat.mean()) / (state_stat.std() + 1e-6), -10., 10.)
        state = torch.FloatTensor(state_np)

        done = False
        episode_score = 0
        while not done:
            action = get_train_action(actor_old, state)
            plotter.neptune_plot({"action": action.item()})
            next_state, reward, done, info = env.step(action)

            states.append(state.detach().squeeze().numpy())
            actions.append(action.item())
            rewards.append(reward.item())
            dones.append(done.item())
            next_states.append(next_state.detach().squeeze().numpy())

            state = next_state

            state_np = state.detach().squeeze().numpy()
            state_stat.update(state_np)
            state_np = np.clip((state_np - state_stat.mean()) / (state_stat.std() + 1e-6), -10., 10.)
            state = torch.FloatTensor(state_np)

            episode_score += reward.item()

        episode_scores.append(episode_score)
        n_episodes += 1
        plotter.neptune_plot({"episode_score": episode_score})

    print(f'\r(episodes {n_episodes}, steps {len(rewards)}), average score: {np.mean(episode_scores)} {episode_scores}')

    scores.append(np.mean(episode_scores))
    scores_avg.append(scores_avg[-1] * 0.9 + np.mean(episode_scores) * 0.1)
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards) / n_episodes
    dones = np.array(dones)
    next_states = np.array(next_states)

    return states, actions, rewards, dones, next_states, np.mean(episode_scores)


def compute_returns_and_advantages(rewards, dones, critic_values):
    returns = np.zeros(rewards.shape)
    deltas = np.zeros(rewards.shape)
    advantages = np.zeros(rewards.shape)

    prev_return, prev_value, prev_advantage = 0, 0, 0
    for i in reversed(range(rewards.shape[0])):
        final_state_bool = 1 - dones[i]

        returns[i] = rewards[i] + GAMMA * prev_return * final_state_bool
        prev_return = returns[i]

        deltas[i] = rewards[i] + GAMMA * prev_value * final_state_bool - critic_values[i]
        prev_value = critic_values[i]

        advantages[i] = deltas[i] + GAMMA * LAMBDA * prev_advantage * final_state_bool
        prev_advantage = advantages[i]

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)
    advantages_tensor = torch.tensor(advantages).float()
    returns_tensor = torch.tensor(returns).float()

    return returns_tensor, advantages_tensor


def update_critic(states_tensor, returns_tensor):
    critic_values_tensor = critic(states_tensor).squeeze()
    loss_critic = nn.MSELoss()(critic_values_tensor, returns_tensor)
    critic_optim.zero_grad()
    loss_critic.backward()
    critic_optim.step()
    return loss_critic


def update_actor(states_tensor, actions_tensor, advantages_tensor):
    # UPDATE ACTOR
    mean_old, std_old = actor_old(states_tensor)
    action_dist_old = torch.distributions.Normal(mean_old.squeeze(), std_old.squeeze())
    action_log_probs_old = action_dist_old.log_prob(actions_tensor).detach()

    mean, std = actor(states_tensor)
    action_dist = torch.distributions.Normal(mean.squeeze(), std.squeeze())
    action_log_probs = action_dist.log_prob(actions_tensor)

    # UPDATE OLD NET
    for target_param, param in zip(actor_old.parameters(), actor.parameters()):
        target_param.data.copy_(param.data)
    # soft_update(actor_old, actor, TAU)
    # if i_update % 5 == 0:
    #     actor_old.load_state_dict(actor.state_dict())

    ratio_of_probs = torch.exp(action_log_probs - action_log_probs_old)
    surrogate1 = ratio_of_probs * advantages_tensor
    surrogate2 = torch.clamp(ratio_of_probs, 1 - EPSILON, 1 + EPSILON) * advantages_tensor
    loss_actor = - torch.min(surrogate1, surrogate2)

    # ADD ENTROPY TERM
    actor_dist_entropy = action_dist.entropy().detach()
    loss_actor = torch.mean(loss_actor - 1e-2 * actor_dist_entropy)
    # loss_actor = loss_actor - 1e-2 * actor_dist_entropy

    actor_optim.zero_grad()
    loss_actor.backward()
    # actor_list_of_grad = [torch.max(torch.abs(param.grad)).item() for param in actor.parameters()]
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 40)
    actor_optim.step()

    return mean, std, loss_actor


def soft_update(target, source, tau):
    # for target_param, param in zip(target_critic.parameters(), critic.parameters()):
    #     target_param.data.copy_(POLYAK * target_param.data + (1.0 - POLYAK) * param.data)
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def plot_neptune():
    # plotter.neptune_plot({'actor_dist_entropy_mean': actor_dist_entropy.mean().item()})
    # plotter.neptune_plot({'actor_mean': mean.mean().item(), 'actor_std': std.mean().item()})
    # plotter.neptune_plot({'loss_actor': loss_actor.item()})
    # plotter.neptune_plot({'loss_critic': loss_critic.item()})
    # plotter.neptune_plot({'actor_max_grad': max(actor_list_of_grad)})
    # mat1 = plotter.matrix_get_prev('actor')
    # mat2 = matrix_get(actor)
    # mse_actor = matrix_mse_mats(mat1, mat2)
    # plotter.neptune_plot({'mse_actor': mse_actor})
    # plotter.matrix_update('critic', critic)
    # plotter.matrix_update('actor', actor)
    pass


def plot_graphs(actor_mean, actor_std, loss, loss_critic, i,
                actor_output_tensor, observations_tensor, critic_output_tensor,
                scores, avg_scores, state_stat_mean, state_stat_std):
    # PLOT
    # mean_list.append(actor_output_tensor.mean().detach().squeeze().item())
    mean_list.append(actor_mean.mean().detach().squeeze().item())
    std_list.append(actor_std.mean().detach().squeeze().item())
    loss_list_actor.append(loss.item())
    loss_list_critic.append(loss_critic.item())
    list_state_mean_1.append(state_stat_mean[0])
    list_state_mean_2.append(state_stat_mean[1])
    list_state_std_1.append(state_stat_std[0])
    list_state_std_2.append(state_stat_std[1])

    if i % PLOT_PER == 0:
        # AX 1
        ax_1.cla()
        input_values_np = observations_tensor.squeeze().numpy()
        x = input_values_np[:, 0]
        y = input_values_np[:, 1]

        actor_output_tensor_np = actor_output_tensor.detach().squeeze().numpy()
        ax_1.scatter(x, y, actor_output_tensor_np, marker='.', alpha=0.09, label='actions')
        # critic_output_tensor_np = critic_output_tensor.detach().squeeze().numpy()
        # ax_1.scatter(x, y, critic_output_tensor_np, marker='.', alpha=0.1, label='critic values')
        ax_1.set_title('Outputs of NN')
        ax_1.legend()

        # AX 2
        ax_2.cla()
        ax_2.plot(mean_list, label='mean')
        ax_2.plot(std_list, label='std')
        ax_2.set_title('Mean & STD')
        ax_2.legend()

        # AX 3
        ax_3.cla()
        ax_3.plot(loss_list_actor, label='actor')
        ax_3.plot(loss_list_critic, label='critic')
        ax_3.set_title('Loss')
        ax_3.legend()

        # AX 4
        ax_4.cla()
        ax_4.plot(scores, label='scores')
        ax_4.plot(avg_scores, label='avg scores')
        ax_4.set_title('Scores')
        ax_4.legend()

        # AX 5
        ax_5.cla()
        ax_5.plot(list_state_mean_1, label='m1')
        ax_5.plot(list_state_mean_2, label='m2')
        ax_5.plot(list_state_std_1, label='s1')
        ax_5.plot(list_state_std_2, label='s2')
        ax_5.set_title('State stat')
        ax_5.legend()

        plt.pause(0.05)




class running_state:
  def __init__(self, state):
    self.len = 1
    self.running_mean = state
    self.running_std = state ** 2

  def update(self, state):
    self.len += 1
    old_mean = self.running_mean.copy()
    self.running_mean[...] = old_mean + (state - old_mean) / self.len
    self.running_std[...] = self.running_std + (state - old_mean) * (state - self.running_mean)

  def mean(self):
    return self.running_mean

  def std(self):
    return np.sqrt(self.running_std / (self.len - 1))


def save_results(path, model_to_save):
    # SAVE
    if SAVE_RESULTS:
        # SAVING...
        print(f"Saving model...")
        torch.save(model_to_save, path)
    return path


if __name__ == '__main__':
    # --------------------------- # PLOTTER & ENV # -------------------------- #
    PLOT_PER = 1

    plotter = ALGPlotter(plot_life=PLOT_LIVE, plot_neptune=NEPTUNE, name='my_run_ppo',
                         tags=["PPO", "clip_grad_norm", "b(s) = 0", ENV_NAME])
    env = SingleAgentEnv(env_name=ENV_NAME, plotter=plotter)

    # --------------------------- # NETS # -------------------------- #
    critic = CriticNet(obs_size=env.observation_size(), n_actions=env.action_size())
    actor = ActorNet(obs_size=env.observation_size(), n_actions=env.action_size())
    actor_old = ActorNet(obs_size=env.observation_size(), n_actions=env.action_size())
    path_to_save = f'data/actor.pt'
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
    fig = plt.figure(figsize=plt.figaspect(.3))
    fig.suptitle('My Run')
    # ax_1 = fig.add_subplot(1, 1, 1, projection='3d')
    ax_1 = fig.add_subplot(1, 5, 1, projection='3d')
    ax_2 = fig.add_subplot(1, 5, 2)
    ax_3 = fig.add_subplot(1, 5, 3)
    ax_4 = fig.add_subplot(1, 5, 4)
    ax_5 = fig.add_subplot(1, 5, 5)


    mean_list, std_list, loss_list_actor, loss_list_critic = [], [], [], []
    list_state_mean_1, list_state_std_1 = [], []
    list_state_mean_2, list_state_std_2 = [], []
    # ---------------------------------------------------------------- #
    # ---------------------------------------------------------------- #

    # Main Process
    train()


    # Example Plays
    plotter.info('Example run...')
    load_and_play(env, 5, path_to_save)

    # ---------------------------------------------------------------- #
