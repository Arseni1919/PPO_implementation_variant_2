import matplotlib.pyplot as plt
import numpy as np
import torch

from alg_GLOBALS import *


X = np.random.random((100,50)) - 0.5    # Random value between -0.5 and +0.5
Y = np.where(X.mean(axis=1) > 0, 1, 0)  # 1 if X[i] > 0; 0 otherwise

print(f'x: {X} \ny: {Y}')

class SimpleModelSigmoid(nn.Module):
    def __init__(self):
        super(SimpleModelSigmoid, self).__init__()
        self.fc1 = nn.Linear(50, 4)
        self.fc2 = nn.Linear(4, 1)

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
        self.fc2 = nn.Linear(4, 1)

        self.net = nn.Sequential(
            self.fc1,
            nn.Tanh(),
            self.fc2,
            # nn.Sigmoid()
        )

    def forward(self, input_value):
        return self.net(input_value)

x_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(Y)

sigmoid_model = SimpleModelSigmoid()
tanh_model = SimpleModelTanh()

sigmoid_optim = torch.optim.SGD(sigmoid_model.parameters(), lr=0.1)
tanh_optim = torch.optim.SGD(tanh_model.parameters(), lr=0.1)

weights_sigmoid = np.zeros((50, 4))
weights_tanh = np.zeros((50, 4))


for epoc in range(50):
    s_output_tensor = sigmoid_model(x_tensor).squeeze()
    s_loss = nn.MSELoss()(s_output_tensor, y_tensor)
    t_output_tensor = tanh_model(x_tensor).squeeze()
    t_loss = nn.MSELoss()(t_output_tensor, y_tensor)

    sigmoid_optim.zero_grad()
    s_loss.backward()
    sigmoid_optim.step()

    tanh_optim.zero_grad()
    t_loss.backward()
    tanh_optim.step()

    s_item = sigmoid_model.fc2.weight.data.squeeze().numpy()
    weights_sigmoid[epoc] = s_item
    t_item = tanh_model.fc2.weight.data.squeeze().numpy()
    weights_tanh[epoc] = t_item

    print(f'sigmoid loss: {s_loss.item()}, tanh loss: {t_loss.item()}')

# print(sigmoid_model(x_tensor).squeeze().detach().numpy())
# print(tanh_model(x_tensor).squeeze().detach().numpy())

weights_sigmoid = np.array(weights_sigmoid)
weights_tanh = np.array(weights_tanh)
w = (weights_sigmoid[:, 0] - weights_sigmoid[:, 0].mean()) / (weights_sigmoid[:, 0].std() + 1e-4)
# print(w)
# plt.plot(w, label='w1')
plt.plot(weights_sigmoid[:, 0], label='s1')
plt.plot(weights_sigmoid[:, 1], label='s2')
plt.plot(weights_sigmoid[:, 2], label='s3')
plt.plot(weights_sigmoid[:, 3], label='s4')

plt.plot(weights_tanh[:, 0], label='t1', marker='.')
plt.plot(weights_tanh[:, 1], label='t2', marker='.')
plt.plot(weights_tanh[:, 2], label='t3', marker='.')
plt.plot(weights_tanh[:, 3], label='t4', marker='.')

plt.legend()

plt.show()
