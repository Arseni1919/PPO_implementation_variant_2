import torch

from alg_GLOBALS import *


class ActorNet(nn.Module):
    """
    obs_size: observation/state size of the environment
    n_actions: number of discrete actions available in the environment
    # hidden_size: size of hidden layers
    """

    def __init__(self, obs_size: int, n_actions: int):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(obs_size, 64)
        self.fc2 = nn.Linear(64, 526)
        self.fc2_2 = nn.Linear(526, 1024)
        self.fc2_3 = nn.Linear(1024, 1024)
        self.fc2_4 = nn.Linear(1024, 526)
        self.fc3 = nn.Linear(526, 64)
        self.fc4 = nn.Linear(64, 1)
        self.head_mean = nn.Linear(64, n_actions)
        self.head_log_std = nn.Linear(64, n_actions)  # to be always positive number
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
        init.xavier_normal_(self.fc2_2.weight)
        init.xavier_normal_(self.fc2_3.weight)
        init.xavier_normal_(self.fc2_4.weight)
        init.xavier_normal_(self.fc3.weight)
        init.xavier_normal_(self.fc4.weight)
        init.xavier_normal_(self.head_mean.weight)
        init.xavier_normal_(self.head_log_std.weight)

        self.body_net = nn.Sequential(
            self.fc1,
            nn.ELU(),
            self.fc2,
            nn.ELU(),
            self.fc2_2,
            nn.ELU(),
            self.fc2_3,
            nn.ELU(),
            self.fc2_4,
            nn.ELU(),
            self.fc3,
            nn.ELU(),
            self.fc4,
            nn.ELU(),
            # self.fc3,
            # nn.Tanh(),
            # # nn.Sigmoid(),
        )

        self.n_actions = n_actions
        self.obs_size = obs_size
        self.entropy_term = 0

    def forward(self, state):
        # if type(state) is np.ndarray:
        #     state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        state = state.float()
        value = self.body_net(state)
        # action_mean = torch.tanh(self.head_mean(value))
        # action_std = torch.exp(self.head_log_std(value))

        # return action_mean, action_std
        return value


class CriticNet(nn.Module):
    """
    obs_size: observation/state size of the environment
    n_actions: number of discrete actions available in the environment
    # hidden_size: size of hidden layers
    """

    def __init__(self, obs_size: int, n_actions: int, n_agents: int = 1):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(obs_size * n_agents, 64)
        # self.fc2 = nn.Linear(64, obs_size * n_agents)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(obs_size * n_agents + n_actions * n_agents, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 1)
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
        init.xavier_normal_(self.fc3.weight)
        init.xavier_normal_(self.fc4.weight)
        init.xavier_normal_(self.fc5.weight)

        self.obs_net = nn.Sequential(
            self.fc1,
            nn.ELU(),
            # nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            # nn.ReLU(),
            self.fc2,
            nn.ELU(),
            self.fc4,
            nn.ELU(),
            self.fc5,
        )

        self.out_net = nn.Sequential(
            self.fc3,
            nn.ELU(),
            self.fc4,
            nn.ELU(),
            self.fc5,
        )

        self.n_actions = n_actions
        self.obs_size = obs_size
        self.n_agents = n_agents
        self.entropy_term = 0

    def forward(self, state):
        # if type(state) is not torch.Tensor:
        #     # state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        #     state = Variable(torch.tensor(state, requires_grad=True).float().unsqueeze(0))
        #     action = Variable(torch.tensor(action, requires_grad=True).float().unsqueeze(0))

        state = state.float()
        value = self.obs_net(state)
        # cat = torch.cat([obs, action], dim=1)
        # value = self.out_net(cat)

        return value
