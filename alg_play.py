import torch

from alg_GLOBALS import *
from alg_env_wrapper import SingleAgentEnv


def get_action(net, observation):
    action_mean, action_std = net(torch.tensor(observation))
    action_dist = torch.distributions.Normal(action_mean, action_std)
    action = action_dist.sample()
    clipped_action = torch.clamp(action, min=-1, max=1)
    return [clipped_action]


def load_and_play(env_to_play, times, path_to_load_model):
    # Example runs
    model = torch.load(path_to_load_model)
    model.eval()
    play(env_to_play, times, model=model)


def play(env, times: int = 1, model: nn.Module = None, max_steps=-1):
    state = env.reset()
    game = 0
    total_reward = 0
    step = 0
    while game < times:
        if model:
            # action = model(state)
            action = get_action(model, state)
        else:
            action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        # total_reward += reward.item()
        env.render()

        step += 1
        if max_steps != -1 and step > max_steps:
            done = True

        if done:
            state = env.reset()
            game += 1
            step = 0
            # print(f'finished game {game} with a total reward: {total_reward}')
            # total_reward = 0
        else:
            state = next_state
    env.close()


if __name__ == '__main__':
    # torch.save(actor, f'{SAVE_PATH}/actor.pt')
    # torch.save(target_actor, f'{SAVE_PATH}/target_actor.pt')
    actor_model = torch.load(f'{SAVE_PATH}/actor.pt')
    actor_model.eval()
    curr_env = SingleAgentEnv(env_name=ENV_NAME)
    play(curr_env, 10, actor_model)
