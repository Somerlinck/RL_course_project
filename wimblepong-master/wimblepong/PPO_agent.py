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

        # We do not want to select action yet as that will be probablistic.
        self.actor = nn.Sequential(
                init_layer(nn.Linear(state_space, 64)),
                nn.Tanh(),
                init_layer(nn.Linear(64, 64)),
                nn.Tanh(),
                init_layer(nn.Linear(64, action_space)),
                nn.Softmax(dim=-1)
            )

        self.critic = nn.Sequential(
                init_layer(nn.Linear(state_space, 64)),
                nn.Tanh(),
                init_layer(nn.Linear(64, 64)),
                nn.Tanh(),
                init_layer(nn.Linear(64, 1)),
            )

    def forward(self, inputs):
        return self.actor(inputs), self.critic(inputs)

    def evaluate(self, state, action):
        aprobs = self.actor(state)
        dist = Categorical(aprobs)
        a_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        return a_logprobs, torch.squeeze(state_value), dist_entropy


class Agent(object):
    def __init__(self, AC_old, AC):
        self.train_device = "cpu"
        self.policy = AC.to(self.train_device)
        self.old_policy = AC_old.to(self.train_device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(AC.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.K_epochs = 3
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
            loss = - torch.min(surrogate1, surrogate2).mean() + F.mse_loss(returns, state_values) - 0.01*dist_entropy.mean()

            # Update network parameters
            if episode_number % 50 == 0:
                loss.backward()
            if episode_number % 200 == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Update weights of the old policy
        self.old_policy.load_state_dict(self.policy.state_dict())
