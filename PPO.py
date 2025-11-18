import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = nn.Linear(obs_dim, 64)
        self.actor = nn.Linear(64, act_dim)
        self.critic = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.relu(self.fc(x))
        return self.actor(x), self.critic(x)

    def get_action(self, obs):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, value, dist.log_prob(action)

def rollout(env, model, steps):
    states, actions, rewards, dones, logps, values = [], [], [], [], [], []
    state = env.reset()

    for _ in range(steps):
        state = torch.tensor(state, dtype=torch.float32)
        action, value, logp = model.get_action(state)

        next_state, reward, done, _ = env.step(action.item())
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        logps.append(logp)
        values.append(value)
        state = next_state

        if done:
            state = env.reset()
        
    return states, actions, rewards, dones, logps, values

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T = len(rewards)
    advantages = torch.zeros(T)
    returns = torch.zeros(T)
    gae = 0
    next_value = 0

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
        next_value = values[t]
    
    return advantages, returns


def ppo_update(model, optimizer, states, actions, logps_old, returns, advantages, clip_ratio=0.2, epochs=10, batch_size=64):
    dataset_size = len(states)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    for _ in range(epochs):
        idx = torch.randperm(dataset_size)
        for i in range(0, dataset_size, batch_size):
            batch_idx = idx[i:i+batch_size]
            batch_states = torch.stack([states[j] for j in batch_idx])
            batch_actions = torch.stack([actions[j] for j in batch_idx])
            batch_logps_old = torch.stack([logps_old[j] for j in batch_idx])
            batch_returns = returns[batch_idx]
            batch_advantages = advantages[batch_idx]

            logits, values = model(batch_states)
            dist = torch.distributions.Categorical(logits=logits)
            new_logps = dist.log_prob(batch_actions)

            ratio = (new_logps - batch_logps_old).exp()

            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values.squeeze(), batch_returns)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * dist.entropy().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
