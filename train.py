from alg_plotter import ALGPlotter
from alg_env_wrapper import SingleAgentEnv
from alg_nets import *
from alg_replay_buffer import ReplayBuffer
from play import load_and_play, play
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
        rewards_to_go, critic_values, advantages = [], [], []
        for traj in minibatch:
            observations, clipped_actions, rewards, dones, new_observations = zip(*traj)
            i_rewards_to_go = compute_rewards_to_go(rewards)
            rewards_to_go.append(i_rewards_to_go)
            i_critic_values = [critic(observation) for observation in observations]
            critic_values.append(i_critic_values)

            # COMPUTE ADVANTAGES
            i_advantages = i_rewards_to_go - i_critic_values
            advantages.append(i_advantages)

        # UPDATE ACTOR
        actor_optim.zero_grad()
        actor_loss = - critic(b_observations, actor(b_observations)).mean()
        actor_loss.backward()
        actor_optim.step()

        # UPDATE CRITIC
        loss = nn.MSELoss()
        critic_optim.zero_grad()
        critic_loss_input = critic(state=b_observations, action=b_actions).squeeze()
        critic_loss = F.mse_loss(critic_loss_input, y)
        critic_loss.backward()
        critic_optim.step()

        # PLOTTER
        plotter.neptune_plot({'loss_critic': critic_loss.item(), 'loss_actor': actor_loss.item()})
        mse_critic = matrix_mse_mats(plotter.matrix_get_prev('critic'), matrix_get(critic))
        plotter.neptune_plot({'mse_critic': mse_critic})
        mat1 = plotter.matrix_get_prev('actor')
        mat2 = matrix_get(actor)
        mse_actor = matrix_mse_mats(mat1, mat2)
        plotter.neptune_plot({'mse_actor': mse_actor})
        plotter.neptune_plot({'max_diff_actor': np.max(np.abs(mat1 - mat2))})

        # RENDER
        if i_update % 4 == 0:
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
    Vals = np.zeros_like(rewards)
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
        action_before_noise = actor(observation)
        # noise_part = torch.normal(mean=torch.tensor(0.0), std=torch.tensor(current_sigma))
        noise_part = next(ou_noise)
        action = action_before_noise * p + noise_part * (1 - p)
        clipped_action = torch.clamp(action, min=-1, max=1)
        new_observation, reward, done, info = env.step(clipped_action)
        trajectory.append((observation, clipped_action, reward, done, new_observation))
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
