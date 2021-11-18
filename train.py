import torch

from alg_plotter import ALGPlotter
from alg_env_wrapper import SingleAgentEnv
from alg_nets import *
from alg_replay_buffer import ReplayBuffer
from play import load_and_play, play, get_action
from alg_functions import *


def train():
    plotter.info('Training...')

    # --------------------------- # MAIN LOOP # -------------------------- #
    for i_update in range(N_UPDATES):
        plotter.info(f'Update {i_update + 1}')
        minibatch = []
        p = i_update / N_UPDATES

        # COLLECT SET OF TRAJECTORIES
        for i_episode in range(N_EPISODES_PER_UPDATE):
            print(f'\r(episode {i_episode + 1})', end='' if i_episode != N_EPISODES_PER_UPDATE - 1 else '\n')
            trajectory = get_trajectory(p)
            minibatch.append(trajectory)

        # COMPUTE REWARDS-TO-GO
        rewards_to_go, critic_values, advantages, observations, actions = [], [], [], [], []
        for traj in minibatch:
            i_observations, i_actions, i_rewards, i_dones, i_new_observations = zip(*traj)
            i_rewards_to_go = compute_rewards_to_go(i_rewards)
            rewards_to_go.append(i_rewards_to_go)
            i_critic_values = [critic(observation, action).detach() for observation, action in zip(i_observations, i_actions)]
            critic_values.append(i_critic_values)
            i_observations = torch.stack(i_observations).squeeze()
            observations.append(i_observations)
            i_actions = torch.stack(i_actions).squeeze()
            actions.append(i_actions)

            # COMPUTE ADVANTAGES
            i_rewards_to_go = torch.stack(i_rewards_to_go).squeeze()
            i_critic_values = torch.stack(i_critic_values).squeeze()
            i_advantages = i_rewards_to_go - i_critic_values
            advantages.append(i_advantages)

        # CONCATENATE ALL
        advantages = torch.cat(advantages)
        observations = torch.cat(observations)
        actions = torch.cat(actions)

        # UPDATE ACTOR
        mean_old, std_old = actor_old(observations)
        action_dist_old = torch.distributions.Normal(mean_old.squeeze(), std_old.squeeze())
        action_log_probs_old = action_dist_old.log_prob(actions)

        mean, std = actor(observations)
        action_dist = torch.distributions.Normal(mean.squeeze(), std.squeeze())
        action_log_probs = action_dist.log_prob(actions)

        ratio_of_probs = torch.exp(action_log_probs - action_log_probs_old)
        surrogate1 = ratio_of_probs * advantages
        surrogate2 = torch.clamp(ratio_of_probs, 1 - EPSILON, 1 + EPSILON) * advantages
        loss_actor = - torch.min(surrogate1, surrogate2)

        # ADD ENTROPY TERM
        actor_dist_entropy = action_dist.entropy()
        loss_actor = loss_actor - actor_dist_entropy
        loss_actor = loss_actor.mean()

        actor_optim.zero_grad()
        loss_actor.backward()
        # torch.nn.utils.clip_grad_norm(actor.parameters(), 40)
        actor_optim.step()

        # UPDATE CRITIC
        # loss = nn.MSELoss()
        # critic_optim.zero_grad()
        # critic_loss_input = critic(state=b_observations, action=b_actions).squeeze()
        # critic_loss = F.mse_loss(critic_loss_input, y)
        # critic_loss.backward()
        # critic_optim.step()

        # UPDATE OLD NET
        actor_old.load_state_dict(actor.state_dict())

        # PLOTTER
        # plotter.neptune_plot({'loss_critic': critic_loss.item(), 'loss_actor': actor_loss.item()})
        # mse_critic = matrix_mse_mats(plotter.matrix_get_prev('critic'), matrix_get(critic))
        # plotter.neptune_plot({'mse_critic': mse_critic})
        # mat1 = plotter.matrix_get_prev('actor')
        # mat2 = matrix_get(actor)
        # mse_actor = matrix_mse_mats(mat1, mat2)
        # plotter.neptune_plot({'mse_actor': mse_actor})
        # plotter.neptune_plot({'max_diff_actor': np.max(np.abs(mat1 - mat2))})

        # RENDER
        if i_update % 4 == 0 and i_update > 0:
            play(env, 1, actor)

    # ---------------------------------------------------------------- #

    # FINISH TRAINING
    plotter.close()
    env.close()
    plotter.info('Finished train.')


def compute_critic_values(observations):
    i_critic_values = [critic(observation) for observation in observations]
    return i_critic_values


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

    while not done:
        action = get_action(actor, observation)
        new_observation, reward, done, info = env.step(action)
        trajectory.append((observation, action, reward, done, new_observation))
        observation = new_observation

    return trajectory


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
    plotter = ALGPlotter(plot_life=PLOT_LIVE, plot_neptune=NEPTUNE, name='my_run', tags=[ENV_NAME])
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
