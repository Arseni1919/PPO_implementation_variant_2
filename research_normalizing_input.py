import matplotlib.pyplot as plt
import numpy as np
import torch

from alg_GLOBALS import *


class SimpleModelSigmoid(nn.Module):
    def __init__(self):
        super(SimpleModelSigmoid, self).__init__()
        self.fc1 = nn.Linear(50, 4)
        self.fc2 = nn.Linear(4, 2)

        self.net = nn.Sequential(
            self.fc1,
            nn.Sigmoid(),
            self.fc2,
            # nn.Sigmoid()
        )

    def forward(self, input_value):
        return self.net(input_value)


class SimpleModelTanh(nn.Module):
    def __init__(self):
        super(SimpleModelTanh, self).__init__()
        self.fc1 = nn.Linear(50, 4)
        self.fc2 = nn.Linear(4, 2)

        self.net = nn.Sequential(
            self.fc1,
            nn.Tanh(),
            self.fc2,
            # nn.Sigmoid()
        )

    def forward(self, input_value):
        return self.net(input_value)


if __name__ == '__main__':
    SEED = 111
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    X = np.random.random((100, 50)) - 0.5  # Random value between -0.5 and +0.5
    Y = np.where(X.mean(axis=1) > 0, 1, 0)  # 1 if X[i] > 0; 0 otherwise

    print(f'x: {X} \ny: {Y}')

    x_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(Y)

    sigmoid_model = SimpleModelSigmoid()
    tanh_model = SimpleModelTanh()
    tanh_model.fc1.load_state_dict(sigmoid_model.fc1.state_dict())
    tanh_model.fc2.load_state_dict(sigmoid_model.fc2.state_dict())

    LR = 4
    sigmoid_optim = torch.optim.SGD(sigmoid_model.parameters(), lr=LR)
    tanh_optim = torch.optim.SGD(tanh_model.parameters(), lr=LR)

    weights_sigmoid = np.zeros((50, 4))
    weights_tanh = np.zeros((50, 4))

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()

    for epoc in range(50):
        s_output_tensor = sigmoid_model(x_tensor).squeeze()
        s_loss = criterion(s_output_tensor, y_tensor.long())
        t_output_tensor = tanh_model(x_tensor).squeeze()
        t_loss = criterion(t_output_tensor, y_tensor.long())

        sigmoid_optim.zero_grad()
        s_loss.backward()
        sigmoid_optim.step()

        tanh_optim.zero_grad()
        t_loss.backward()
        tanh_optim.step()

        s_item = sigmoid_model.fc2.weight.data.squeeze().numpy()
        weights_sigmoid[epoc] = s_item[0, :]
        t_item = tanh_model.fc2.weight.data.squeeze().numpy()
        weights_tanh[epoc] = t_item[0, :]

        print(f'sigmoid loss: {s_loss.item()}, tanh loss: {t_loss.item()}')

    # print(sigmoid_model(x_tensor).squeeze().detach().numpy())
    # print(tanh_model(x_tensor).squeeze().detach().numpy())

    weights_sigmoid = np.array(weights_sigmoid)
    weights_tanh = np.array(weights_tanh)
    w = (weights_sigmoid[:, 0] - weights_sigmoid[:, 0].mean()) / (weights_sigmoid[:, 0].std() + 1e-4)
    # print(w)
    # plt.plot(w, label='w1')

    fig = plt.figure(figsize=plt.figaspect(.5))
    fig.suptitle('Analysis of Normalizing Input')
    # ax_1 = fig.add_subplot(1, 1, 1, projection='3d')
    ax_1 = fig.add_subplot(1, 2, 1)
    ax_2 = fig.add_subplot(1, 2, 2)

    ax_1.plot(weights_sigmoid[:, 0], label='s1')
    ax_1.plot(weights_sigmoid[:, 1], label='s2')
    ax_1.plot(weights_sigmoid[:, 2], label='s3')
    ax_1.plot(weights_sigmoid[:, 3], label='s4')
    ax_1.set_title('Sigmoid')
    ax_1.legend()

    ax_2.plot(weights_tanh[:, 0], label='t1', marker='.')
    ax_2.plot(weights_tanh[:, 1], label='t2', marker='.')
    ax_2.plot(weights_tanh[:, 2], label='t3', marker='.')
    ax_2.plot(weights_tanh[:, 3], label='t4', marker='.')
    ax_2.set_title('Tanh')
    ax_2.legend()

    plt.show()

