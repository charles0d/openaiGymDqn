import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DeepQNet(nn.Module):
    def __init__(self, lr, input_dims, hidden1_dims, hidden2_dims, n_actions):
        super(DeepQNet, self).__init__()
        self.input_dims = input_dims
        self.hidden1_dims = hidden1_dims
        self.hidden2_dims = hidden2_dims
        self.n_actions = n_actions

        self.layer1 = nn.Linear(self.input_dims, self.hidden1_dims)
        self.layer2 = nn.Linear(self.hidden1_dims, self.hidden2_dims)
        self.out_layer = nn.Linear(self.hidden2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return self.out_layer(x)
