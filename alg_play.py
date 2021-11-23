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


def play(env, times: int = 1, model: nn.Module = None, max_steps=-1):
    state = env.reset()
    state_stat = running_state(env.reset().detach().squeeze().numpy())
    game = 0
    total_reward = 0
    step = 0
    while game < times:
        if model:
            # action = model(state)

            state_np = state.detach().squeeze().numpy()
            state_stat.update(state_np)
            state_np = np.clip((state_np - state_stat.mean()) / (state_stat.std() + 1e-6), -10., 10.)
            state = torch.FloatTensor(state_np)

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
    actor_model = torch.load(f'data/actor.pt')
    actor_model.eval()
    curr_env = SingleAgentEnv(env_name=ENV_NAME)
    play(curr_env, 10, actor_model)
