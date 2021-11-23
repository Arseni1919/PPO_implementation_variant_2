from alg_play import play, load_and_play
from alg_env_wrapper import SingleAgentEnv
import argparse
import gym
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from drawnow import drawnow
import matplotlib.pyplot as plt


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


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_mean = nn.Linear(64, 1)
        self.fc_log_std = nn.Linear(64, 1)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        action_mean = self.fc_mean(x)
        action_std = torch.exp(self.fc_log_std(x))
        return action_mean.squeeze(), action_std.squeeze()


def get_action(state):
    action_mean, action_std = actor(state)
    action_dist = torch.distributions.Normal(action_mean, action_std)
    action = action_dist.sample()
    return action.item()


def synchronize_actors():
    for target_param, param in zip(actor_old.parameters(), actor.parameters()):
        target_param.data.copy_(param.data)


def update_actor(state, action, advantage):
    mean_old, std_old = actor_old(state)
    action_dist_old = torch.distributions.Normal(mean_old, std_old)
    action_log_probs_old = action_dist_old.log_prob(action)

    mean, std = actor(state)
    action_dist = torch.distributions.Normal(mean, std)
    action_log_probs = action_dist.log_prob(action)

    # update old actor before update current actor
    synchronize_actors()

    r_theta = torch.exp(action_log_probs - action_log_probs_old)
    surrogate1 = r_theta * advantage
    surrogate2 = torch.clamp(r_theta, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon) * advantage
    loss = -torch.min(surrogate1, surrogate2).mean()

    entropy = action_dist.entropy()
    loss = torch.mean(loss - 1e-2 * entropy)

    actor_optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm(actor.parameters(), 40)
    actor_optimizer.step()
    return mean, std, loss


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        value = self.fc3(x)
        return value.squeeze()


def update_critic(state, target):
    state_value = critic(state)
    loss = F.mse_loss(state_value, target)
    critic_optimizer.zero_grad()
    loss.backward()
    critic_optimizer.step()
    return state_value, loss


def main():
    state = env.reset()
    state_stat = running_state(state)

    for j in range(cfg.max_episode):
        start_time = time.perf_counter()
        episode_score = 0
        episode = 0
        memory = []

        with torch.no_grad():

            # SAMPLE TRAJECTORIES
            while len(memory) < cfg.batch_size:
                episode += 1
                small_score = 0
                state = env.reset()
                state_stat.update(state)
                state = np.clip((state - state_stat.mean()) / (state_stat.std() + 1e-6), -10., 10.)

                for s in range(1000):
                    action = get_action(torch.tensor(state).float()[None, :])
                    next_state, reward, done, _ = env.step([action])

                    state_stat.update(next_state)
                    next_state = np.clip((next_state - state_stat.mean()) / (state_stat.std() + 1e-6), -10., 10.)
                    memory.append([state, action, reward, next_state, done])

                    state = next_state
                    episode_score += reward
                    small_score += reward

                    if done:
                        # print(f'### in small episode {episode} the score is {small_score}')
                        break

            state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(
                lambda x: np.array(x).astype(np.float32), zip(*memory)
            )

            state_batch = torch.tensor(state_batch).float()
            values = critic(state).detach().cpu().numpy()

            # COMPUTE RETURNS AND ADVANTAGES
            returns = np.zeros(action_batch.shape)
            deltas = np.zeros(action_batch.shape)
            advantages = np.zeros(action_batch.shape)

            prev_return = 0
            prev_value = 0
            prev_advantage = 0
            for i in reversed(range(reward_batch.shape[0])):
                returns[i] = reward_batch[i] + cfg.gamma * prev_return * (1 - done_batch[i])
                # generalized advantage estimation
                deltas[i] = reward_batch[i] + cfg.gamma * prev_value * (1 - done_batch[i]) - values[i]
                advantages[i] = deltas[i] + cfg.gamma * cfg.gae_lambda * prev_advantage * (1 - done_batch[i])

                prev_return = returns[i]
                prev_value = values[i]
                prev_advantage = advantages[i]

            advantages = (advantages - advantages.mean()) / advantages.std()

            advantages = torch.tensor(advantages).float()
            action_batch = torch.tensor(action_batch).float()
            returns = torch.tensor(returns).float()

        # UPDATE CRITIC
        critic_output, loss_critic = update_critic(state_batch, returns)

        # UPDATE ACTOR
        actor_mean, actor_std, loss_actor = update_actor(state_batch, action_batch, advantages)

        # PLOT
        episode_score /= episode
        print(f'# [{j}] last_score {episode_score:5f}, steps {len(memory)}, ({time.perf_counter() - start_time:2f} sec/eps)')
        avg_score_plot.append(avg_score_plot[-1] * 0.99 + episode_score * 0.01)
        last_score_plot.append(episode_score)
        # drawnow(draw_fig)
        plot_graphs(actor_mean, actor_std, loss_actor.item(), loss_critic.item(), i, action_batch, state_batch,
                    critic_output, last_score_plot, avg_score_plot)

    env.close()


def save_results(model_to_save, name):
    path_to_save = f'data/{name}.pt'
    # SAVING...
    print(f"Saving {name}'s model...")
    torch.save(model_to_save, path_to_save)
    return path_to_save


def plot_graphs(actor_mean, actor_std, loss_actor, loss_critic, i, actor_output_tensor, observations_tensor, critic_output_tensor, scores, avg_scores):
    # PLOT
    # mean_list.append(actor_output_tensor.mean().detach().squeeze().item())
    mean_list.append(actor_mean.mean().detach().squeeze().item())
    std_list.append(actor_std.mean().detach().squeeze().item())
    loss_list_actor.append(loss_actor)
    loss_list_critic.append(loss_critic)

    if i % PLOT_PER == 0:
        # AX 1
        ax_1.cla()
        input_values_np = observations_tensor.detach().squeeze().numpy()
        x = input_values_np[:, 0]
        y = input_values_np[:, 1]

        actor_output_tensor_np = actor_output_tensor.detach().squeeze().numpy()
        ax_1.scatter(x, y, actor_output_tensor_np, marker='.', alpha=0.1, label='actions')
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

        plt.pause(0.05)


if __name__ == '__main__':
    last_score_plot = [-100]
    avg_score_plot = [-100]

    parser = argparse.ArgumentParser(description='PyTorch PPO solution of MountainCarContinuous-v0')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--actor_lr', type=float, default=1e-3)
    parser.add_argument('--critic_lr', type=float, default=1e-3)
    parser.add_argument('--clip_epsilon', type=float, default=0.2)
    parser.add_argument('--gae_lambda', type=float, default=0.97)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--max_episode', type=int, default=30)
    cfg = parser.parse_args()
    ENV_NAME = 'MountainCarContinuous-v0'
    env = gym.make(ENV_NAME)

    actor = Actor()
    actor_old = Actor()
    critic = Critic()
    actor_optimizer = optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    # FOR PLOTS
    PLOT_PER = 2
    # fig = plt.figure()
    fig = plt.figure(figsize=plt.figaspect(.3))
    fig.suptitle('Example 1 Run')
    ax_1 = fig.add_subplot(1, 4, 1, projection='3d')
    ax_2 = fig.add_subplot(1, 4, 2)
    ax_3 = fig.add_subplot(1, 4, 3)
    ax_4 = fig.add_subplot(1, 4, 4)
    mean_list, std_list, loss_list_actor, loss_list_critic = [], [], [], []

    main()
    plt.pause(0)
    plt.close()

    # Save
    path_actor_model = save_results(actor, name='actor_example_1')

    # Example Plays
    print('Example run...')
    wrapped_env = SingleAgentEnv(env_name=ENV_NAME)
    load_and_play(wrapped_env, 3, path_actor_model)


# def draw_fig():
#     plt.title('reward')
#     plt.plot(last_score_plot, '-')
#     plt.plot(avg_score_plot, 'r-')

# def get_state_value(state):
#     state_value = critic(state)
#     return state_value
