import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F


def init_layer(m):
    weight = m.weight.data
    weight.normal_(0, 1)
    weight *= 1.0 / torch.sqrt(weight.pow(2).sum(1, keepdim=True))
    nn.init.constant_(m.bias.data, 0)
    return m


class ActorCritic(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()

        self.conv = torch.nn.Sequential(
            nn.Conv2d(3, 32, 8),  # _x_x32
            nn.ReLU(),
            nn.MaxPool2d(4),  # 98x98x16
            nn.Conv2d(32, 64, 4),  # _x_x16
            nn.ReLU(),
            torch.nn.MaxPool2d(4),  # 48x48x16
            torch.nn.Conv2d(64, 64, 3),  # 46x46x8
            nn.ReLU(),
            torch.nn.MaxPool2d(4)
        )

        # We do not want to select action yet as that will be probablistic.
        self.actor = nn.Sequential(
                init_layer(nn.Linear(256, 64)),
                nn.Tanh(),
                init_layer(nn.Linear(64, 64)),
                nn.Tanh(),
                init_layer(nn.Linear(64, action_space)),
                nn.Softmax(dim=-1)
            )

        self.critic = nn.Sequential(
                init_layer(nn.Linear(256, 64)),
                nn.Tanh(),
                init_layer(nn.Linear(64, 64)),
                nn.Tanh(),
                init_layer(nn.Linear(64, 1)),
            )

    def forward(self, inputs):
        feature_maps = self.conv(inputs)
        n_proc = feature_maps.shape[0]
        return self.actor(feature_maps.view(n_proc, 256)), self.critic(feature_maps.view(n_proc, 256))

    def evaluate(self, state, action):
        aprobs, state_value = self.forward(state)
        dist = Categorical(aprobs)
        a_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return a_logprobs, torch.squeeze(state_value), dist_entropy


class Agent(object):
    def __init__(self, AC_old, AC):
        self.train_device = "cpu"
        self.policy = AC.to(self.train_device)
        self.old_policy = AC_old.to(self.train_device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(AC.parameters(), lr=3e-4)
        self.gamma = 0.98
        self.K_epochs = 4
        self.epsilon_clip = 0.2

        self.states = []
        self.next_states = []
        self.log_probs = []
        self.rewards = []
        self.actions = []
        self.dones = []

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)
        aprob, _ = self.old_policy.forward(x)
        distribution = Categorical(aprob)

        if evaluation:
            action = aprob.argmax()
        else:
            action = distribution.sample()

        return action, distribution.log_prob(action)

    def store_outcome(self, state_array, log_prob, action, reward, next_state_array, done):
        self.states.append(torch.from_numpy(state_array).float())
        self.next_states.append(torch.from_numpy(next_state_array).float())
        self.log_probs.append(log_prob)
        self.actions.append(action)
        self.rewards.append(torch.Tensor([reward]))
        self.dones.append(torch.Tensor([done]).int())

    def update(self, episode_number):
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        old_states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        old_actions = torch.stack(self.actions, dim=0).to(self.train_device).squeeze(-1)
        old_log_probs = torch.stack(self.log_probs, dim=0).to(self.train_device).squeeze(-1)
        dones = torch.stack(self.dones, dim=0).to(self.train_device).squeeze(-1)
        self.states, self.actions, self.log_probs, self.rewards, self.next_states, self.dones = [], [], [], [], [], []

        # Compute discounted rewards
        _, next_state_values, _ = self.policy.evaluate(next_states, old_actions)
        next_state_values = next_state_values.detach()
        returns = rewards + self.gamma * next_state_values * (1-dones)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluate old actions and values
            log_probs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # pi_theta / pi_theta_old
            ratios = torch.exp(log_probs - old_log_probs.detach())

            # Surrogate Loss
            advantages = returns - state_values.detach()
            surrogate1 = ratios * advantages
            surrogate2 = torch.clamp(ratios, min=1-self.epsilon_clip, max=1+self.epsilon_clip) * advantages
            loss = - torch.min(surrogate1, surrogate2).mean() + F.mse_loss(returns, state_values) - 0.001*dist_entropy.mean()

            # Update network parameters
            if episode_number % 50 == 0:
                loss.backward()
            if episode_number % 200 == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Update weights of the old policy
        self.old_policy.load_state_dict(self.policy.state_dict())
