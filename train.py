from alg_plotter import ALGPlotter
from alg_env_wrapper import SingleAgentEnv
from alg_nets import *
from alg_replay_buffer import ReplayBuffer
from play import play
from alg_functions import *


def train():
    plotter.info('Training...')
    # --------------------------- # FIRST OBSERVATION # -------------------------- #
    observation = env.reset()

    # --------------------------- # FINISH TRAINING # -------------------------- #
    close()
    plotter.info('Finished train.')


def close():
    plotter.close()
    env.close()


def save_results(model_to_save, name):
    path_to_save = f'{SAVE_PATH}/{name}.pt'
    # Save
    if SAVE_RESULTS:
        # Saving...
        plotter.info('Saving results...')
        torch.save(model_to_save, path_to_save)
    return path_to_save


def example_runs(env_to_play, times, path_to_load_model):
    # Example runs
    plotter.info('Example run...')
    model = torch.load(path_to_load_model)
    model.eval()
    play(env_to_play, times, model=model)


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

    # --------------------------- # PLOTTER INIT # -------------------------- #
    plotter.neptune_set_parameters()
    plotter.matrix_update('critic', critic)
    plotter.matrix_update('actor', actor)

    # --------------------------- # REPLAY BUFFER # -------------------------- #
    replay_buffer = ReplayBuffer()

    # --------------------------- # NOISE # -------------------------- #
    current_sigma = SIGMA
    normal_distribution = Normal(torch.tensor(0.0), torch.tensor(current_sigma))

    # Simple Ornstein-Uhlenbeck Noise generator
    ou_noise = OUNoise()

    # ---------------------------------------------------------------- #
    # ---------------------------------------------------------------- #

    # main process
    train()

    # Save
    path = save_results(actor, name='actor')

    # example plays
    example_runs(env, 3, path)

    # ---------------------------------------------------------------- #
