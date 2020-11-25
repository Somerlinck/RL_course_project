import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 128
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2 = torch.nn.Linear(self.hidden, self.action_space)
        self.sigma = torch.nn.Parameter(torch.zeros(1) + 10)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight, std=0.001)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        model = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            self.fc2,
            torch.nn.Softmax(dim=-1)
        )
        return model(x)


class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.log_prob = []
        self.rewards = []

    def episode_finished(self):
        log_prob = torch.stack(self.log_prob, dim=0) \
                .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        self.states, self.log_prob, self.rewards = [], [], []

        # Compute discounted rewards (use the discount_rewards function)
        discounted_rewards = discount_rewards(rewards, self.gamma)
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)
        # Compute the optimization term
        weighted_probs = log_prob * discounted_rewards

        # Compute the gradients of loss w.r.t. network parameters
        loss = -torch.mean(weighted_probs)
        loss.backward()

        # Update network parameters using self.optimizer and zero gradients (T1)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)

        # Pass state x through the policy network
        aprob = self.policy.forward(x)
        c = Categorical(aprob)
        if evaluation:
            action = aprob.argmax()
            print(aprob)
        else:
            action = c.sample()

        return action, c.log_prob(action)

    def store_outcome(self, observation, log_prob, reward):
        self.states.append(observation)
        self.log_prob.append(log_prob)
        self.rewards.append(torch.Tensor([reward]))
