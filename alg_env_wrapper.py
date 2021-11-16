import gym.spaces
import torch

from GLOBALS import *

# from alg_plotter import plotter


class SingleAgentEnv:
    def __init__(self, env_name, plotter=None):
        self.env_name = env_name
        self.plotter = plotter
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        observation = self.env.reset()
        observation = Variable(torch.tensor(observation, requires_grad=True).float().unsqueeze(0))
        return observation

    def render(self):
        self.env.render()

    def sample_action(self):
        action = self.env.action_space.sample()
        action = Variable(torch.tensor(action, requires_grad=True).float().unsqueeze(0))
        return action

    def sample_observation(self):
        observation = self.env.observation_space.sample()
        observation = Variable(torch.tensor(observation, requires_grad=True).float().unsqueeze(0))
        return observation

    def step(self, action):
        action = self.prepare_action(action)
        observation, reward, done, info = self.env.step(action)
        observation = Variable(torch.tensor(observation, requires_grad=True).float().unsqueeze(0))
        reward = Variable(torch.tensor(reward).float().unsqueeze(0))
        done = torch.tensor(done)
        return observation, reward, done, info

    def prepare_action(self, action):
        action = action.detach().squeeze().numpy()
        if self.env_name == "CartPole-v1":
            action = 1 if action > 0.5 else 0
            # print(action)
        elif self.env_name == "MountainCarContinuous-v0":
            action = [action]
        elif self.env_name == "LunarLanderContinuous-v2":
            action = action
        elif self.env_name == "BipedalWalker-v3":
            action = action
        else:
            if self.plotter:
                self.plotter.error('action!')
        return action

    def close(self):
        self.env.close()

    def observation_size(self):
        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            return self.env.observation_space.n
        if isinstance(self.env.observation_space, gym.spaces.Box):
            return self.env.observation_space.shape[0]
        return None

    def action_size(self):
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            # self.env.action_space.n
            return 1
        if isinstance(self.env.action_space, gym.spaces.Box):
            return self.env.action_space.shape[0]
        return None


class MultiAgentEnv:
    def __init__(self):
        pass


# env = SingleAgentEnv(env_name=SINGLE_AGENT_ENV_NAME)
