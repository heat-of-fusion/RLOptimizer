import torch
import torch.nn as nn

from torchsummary import summary

class PolicyNet(nn.Module):
    def __init__(self, state_size, hidden_size = 256, action_size = 5):
        super(PolicyNet, self).__init__()

        self.leaky_relu = nn.LeakyReLU(0.1)
        self.softmax = nn.Softmax(dim = 1)

        self.actor_fc1 = nn.Linear(state_size, hidden_size)
        self.actor_fc2 = nn.Linear(hidden_size, hidden_size)

        self.beta_1 = nn.Linear(hidden_size, action_size)
        self.beta_2 = nn.Linear(hidden_size, action_size)
        self.gamma_3 = nn.Linear(hidden_size, action_size)

        return

    def forward(self, state):
        actor_x = self.leaky_relu(self.actor_fc1(state))
        actor_x = self.leaky_relu(self.actor_fc2(actor_x))

        beta_1 = self.softmax(self.beta_1(actor_x))
        beta_2 = self.softmax(self.beta_2(actor_x))
        gamma_3 = self.softmax(self.gamma_3(actor_x))

        return beta_1, beta_2, gamma_3

class CriticNet(nn.Module):
    def __init__(self, state_size, hidden_size = 256):
        super(CriticNet, self).__init__()

        self.leaky_relu = nn.LeakyReLU(0.1)

        self.critic_fc1 = nn.Linear(state_size, hidden_size)
        self.critic_fc2 = nn.Linear(hidden_size, hidden_size)
        self.critic_fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        critic_x = self.leaky_relu(self.critic_fc1(state))
        critic_x = self.leaky_relu(self.critic_fc2(critic_x))
        critic_out = self.critic_fc3(critic_x)

        return critic_out