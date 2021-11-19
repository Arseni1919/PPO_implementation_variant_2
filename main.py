import torch

from alg_plotter import ALGPlotter
from alg_env_wrapper import SingleAgentEnv
from alg_nets import *
from alg_replay_buffer import ReplayBuffer
from alg_play import play
from alg_functions import *
torch.autograd.set_detect_anomaly(True)

plotter = ALGPlotter(plot_life=PLOT_LIVE, plot_neptune=NEPTUNE, name='my_run', tags=[ENV_NAME])
plotter.neptune_set_parameters()
env = SingleAgentEnv(env_name=ENV_NAME, plotter=plotter)

# --------------------------- # NETS # -------------------------- #
critic = CriticNet(obs_size=env.observation_size(), n_actions=env.action_size(), n_agents=1)
target_critic = CriticNet(obs_size=env.observation_size(), n_actions=env.action_size(), n_agents=1)
target_critic.load_state_dict(critic.state_dict())
actor = ActorNet(obs_size=env.observation_size(), n_actions=env.action_size())
target_actor = ActorNet(obs_size=env.observation_size(), n_actions=env.action_size())
target_actor.load_state_dict(actor.state_dict())

critic_optim = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)
actor_optim = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)

plotter.matrix_update('critic', critic)
plotter.matrix_update('actor', actor)

# --------------------------- # REPLAY BUFFER # -------------------------- #
replay_buffer = ReplayBuffer()

# --------------------------- # NORMAL DISTRIBUTION # -------------------------- #
current_sigma = SIGMA
normal_distribution = Normal(torch.tensor(0.0), torch.tensor(current_sigma))
# normal_distribution.sample()

# Simple Ornstein-Uhlenbeck Noise generator
ou_noise = OUNoise()

# --------------------------- # FIRST OBSERVATION # -------------------------- #
observation = env.reset()

rewards = 0
step, episode, p = 0, 0, 0
while episode < N_EPISODES:  # while step < N_STEPS and episode < N_EPISODES:
    # for step in range(N_STEPS):
    print(f'\r(step {step - REPLAY_BUFFER_SIZE})', end='')
    # --------------------------- # STEP # -------------------------- #
    # action = env.sample_action()  # your agent here (this takes random actions)
    with torch.no_grad():
        action_before_noise = actor(observation)
        # noise_part = torch.normal(mean=torch.tensor(0.0), std=torch.tensor(current_sigma))
        noise_part = next(ou_noise)
        action = action_before_noise * p + noise_part * (1 - p)
        clipped_action = torch.clamp(action, min=-1, max=1)
        new_observation, reward, done, info = env.step(clipped_action)
        if step > WARMUP:
            p = episode / N_EPISODES
            plotter.neptune_plot({'action': clipped_action.item(), 'action_before_noise': action_before_noise.item()})
            plotter.neptune_plot({'action_difference': abs(clipped_action.item() - action_before_noise.item())})
            plotter.neptune_plot({'noise_part': noise_part})
            plotter.neptune_plot({'p': p})

    # --------------------------- # STORE # -------------------------- #
    replay_buffer.append((observation, clipped_action, reward, done, new_observation))
    rewards += reward.item()

    # --------------------------- # UPDATE # -------------------------- #
    observation = new_observation
    if done:
        observation = env.reset()
        plotter.plots_update_data({'rewards': rewards})
        if step > WARMUP:
            plotter.debug(f'episode: {episode}. Total reward: {rewards}')
            plotter.plots_update_data({'rewards': rewards})
            plotter.neptune_plot({'episode_score': rewards})
        episode += 1
        rewards = 0
        ou_noise = OUNoise()

    if step > WARMUP:
        # print(f'step: {step}')
        # --------------------------- # MINIBATCH # -------------------------- #
        minibatch = replay_buffer.sample(n=BATCH_SIZE)
        b_observations, b_actions, b_rewards, b_dones, b_next_observations = zip(*minibatch)
        b_observations = torch.stack(b_observations).squeeze()
        b_actions = torch.stack(b_actions).squeeze(1)
        b_rewards = torch.stack(b_rewards).squeeze()
        b_dones = torch.stack(b_dones).squeeze()
        b_next_observations = torch.stack(b_next_observations).squeeze()

        # --------------------------- # Y # -------------------------- #
        with torch.no_grad():
            next_q = target_critic(state=b_next_observations, action=target_actor(b_next_observations)).squeeze()
            next_q = (~b_dones) * next_q
            y = b_rewards + GAMMA * next_q

        # --------------------------- # UPDATE CRITIC # -------------------------- #
        loss = nn.MSELoss()
        critic_optim.zero_grad()
        critic_loss_input = critic(state=b_observations, action=b_actions).squeeze()
        critic_loss = F.mse_loss(critic_loss_input, y)
        critic_loss.backward()
        critic_optim.step()

        # --------------------------- # UPDATE ACTOR # -------------------------- #
        actor_optim.zero_grad()
        actor_loss = - critic(b_observations, actor(b_observations)).mean()
        actor_loss.backward()
        actor_optim.step()

        # --------------------------- # UPDATE TARGET NETS # -------------------------- #
        soft_update(target_critic, critic, TAU)
        soft_update(target_actor, actor, TAU)

        # --------------------------- # PLOTTER # -------------------------- #
        plotter.neptune_plot({'loss_critic': critic_loss.item(), 'loss_actor': actor_loss.item()})
        mse_critic = matrix_mse_mats(plotter.matrix_get_prev('critic'), matrix_get(critic))
        plotter.neptune_plot({'mse_critic': mse_critic})
        mat1 = plotter.matrix_get_prev('actor')
        mat2 = matrix_get(actor)
        mse_actor = matrix_mse_mats(mat1, mat2)
        plotter.neptune_plot({'mse_actor': mse_actor})
        plotter.neptune_plot({'max_diff_actor': np.max(np.abs(mat1-mat2))})

        # --------------------------- # RENDER # -------------------------- #
        if episode > 65:
            env.render()

        if step % 100 == 0 and episode % 10 == 0:
            # plotter.plots_online()
            # plotter.plot_nn_map(net_actor=actor, net_critic=critic)
            # print(f'mse_critic: {mse_critic}, mse_actor: {mse_actor}')
            pass

        plotter.matrix_update('critic', critic)
        plotter.matrix_update('actor', actor)
        # ---------------------------------------------------------------- #
    step += 1

plotter.close()
env.close()
plotter.info('Finished train.')

# Save & Run
if SAVE_RESULTS:

    # Saving...
    plotter.info('Saving results...')
    torch.save(actor, f'{SAVE_PATH}/actor.pt')
    torch.save(target_actor, f'{SAVE_PATH}/target_actor.pt')

    # Example runs
    plotter.info('Example run...')
    model = torch.load(f'{SAVE_PATH}/target_actor.pt')
    model.eval()
    play(env, 3, model=model)

