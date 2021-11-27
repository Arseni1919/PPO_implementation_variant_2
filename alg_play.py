import torch

from alg_GLOBALS import *
from alg_env_wrapper import SingleAgentEnv


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
        action_std = torch.exp(self.fc_log_std(x))  # to be always positive number
        return action_mean.squeeze(), action_std.squeeze()


def get_action(net, observation):
    if not isinstance(observation, torch.Tensor):
        observation = torch.tensor(observation)
    action_mean, action_std = net(observation)
    action_dist = torch.distributions.Normal(action_mean, action_std)
    action = action_dist.sample()
    # clipped_action = torch.clamp(action, min=-1, max=1)
    return action


def load_and_play(env_to_play, times, path_to_load_model):
    # Example runs
    model = torch.load(path_to_load_model)
    model.eval()
    play(env_to_play, times, model=model)


# class RunningStateStat:
#     def __init__(self, state):
#         self.len = 1
#         self.running_mean = state
#         self.running_std = state ** 2
#
#     def update(self, state):
#         self.len += 1
#         old_mean = self.running_mean.copy()
#         self.running_mean[...] = old_mean + (state - old_mean) / self.len
#         self.running_std[...] = self.running_std + (state - old_mean) * (state - self.running_mean)
#
#     def mean(self):
#         return self.running_mean
#
#     def std(self):
#         return np.sqrt(self.running_std / (self.len - 1))
#
#     def get_normalized_state(self, state_tensor):
#         state_np = state_tensor.detach().squeeze().numpy()
#         self.update(state_np)
#         state_np = np.clip((state_np - self.mean()) / (self.std() + 1e-6), -10., 10.)
#         output_state_tensor = torch.FloatTensor(state_np)
#         return output_state_tensor


def play(env, times: int = 1, model: nn.Module = None, max_steps=-1):
    state = env.reset()
    # state_stat = RunningStateStat(env.reset().detach().squeeze().numpy())
    game = 0
    total_reward = 0
    step = 0
    while game < times:
        if model:
            # action = model(state)
            # state = state_stat.get_normalized_state(state)

            action = get_action(model, state)
        else:
            action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        total_reward += reward.item()
        env.render()

        step += 1
        if max_steps != -1 and step > max_steps:
            done = True

        if done:
            state = env.reset()
            game += 1
            step = 0
            print(f'finished game {game} with a total reward: {total_reward}')
            total_reward = 0
        else:
            state = next_state
    env.close()


if __name__ == '__main__':
    # torch.save(actor, f'{SAVE_PATH}/actor.pt')
    # torch.save(target_actor, f'{SAVE_PATH}/target_actor.pt')
    # actor_model = torch.load(f'data/actor_example_1.pt')

    # ENV_NAME = "MountainCarContinuous-v0"
    # ENV_NAME = "CartPole-v1"
    # ENV_NAME = 'LunarLanderContinuous-v2'
    ENV_NAME = "BipedalWalker-v3"

    path_to_load = f'data/actor_{ENV_NAME}.pt'

    if ENV_NAME in ['BipedalWalker-v3']:
        load_dict = torch.load(path_to_load)
        actor_model = load_dict['model']
        actor_model.eval()
        curr_env = SingleAgentEnv(env_name=ENV_NAME)
        curr_env.state_stat.running_mean = load_dict['mean']
        curr_env.state_stat.running_std = load_dict['std']
        curr_env.state_stat.len = load_dict['len']
    else:
        actor_model = torch.load(path_to_load)
        actor_model.eval()
        curr_env = SingleAgentEnv(env_name=ENV_NAME)

    play(curr_env, 20, actor_model)
