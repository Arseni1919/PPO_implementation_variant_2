from alg_plotter import ALGPlotter
from alg_env_wrapper import SingleAgentEnv
from alg_nets import *
from alg_replay_buffer import ReplayBuffer
from play import load_and_play
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
        # TODO

        # COMPUTE ADVANTAGES
        # TODO

        # UPDATE ACTOR
        # TODO

        # UPDATE CRITIC
        # TODO

        # PLOTTER
        # TODO

        # RENDER
        # TODO

        pass
    # ---------------------------------------------------------------- #

    # FINISH TRAINING
    plotter.close()
    env.close()
    plotter.info('Finished train.')


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
