# adapted from cleanrl ppo implementation
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from networks.multigrid_network import MultiGridNetwork

NUM_DIRECTIONS = 4


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# the actual neural network (actor + critic)
class Agent(nn.Module):
    def __init__(self, config, obs, n_actions, n_agents, agent_id, device):
        super().__init__()
        self.config = config
        self.device = device

        # use multigrid network as the actor since it handles image + direction obs
        self.actor = MultiGridNetwork(obs, config, n_actions, n_agents, agent_id)

        # critic just needs to output a single value
        feature_size = 64 + config.fc_direction
        self.critic = nn.Sequential(
            layer_init(nn.Linear(feature_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def _get_features(self, obs):
        # run image through conv layers
        x = torch.tensor(obs['image']).to(self.device)
        x = self.actor.process_image(x)
        batch_dim = x.shape[0]
        image_features = self.actor.image_layers(x).reshape(batch_dim, -1)

        # run direction through its layers
        dirs = torch.tensor(obs['direction']).to(self.device)
        if batch_dim == 1:
            dirs = dirs.unsqueeze(0)
        dirs_onehot = torch.nn.functional.one_hot(
            dirs.to(torch.int64), num_classes=NUM_DIRECTIONS
        ).reshape((batch_dim, -1)).float()
        dir_features = self.actor.direction_layers(dirs_onehot)

        return torch.cat([image_features, dir_features], dim=-1)

    def get_value(self, obs):
        features = self._get_features(obs)
        return self.critic(features)

    def get_action_and_value(self, obs, action=None):
        features = self._get_features(obs)
        logits = self.actor.head(features)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(features)


# wrapper around Agent that handles training logic, buffers, etc
class PPOAgent:
    def __init__(self, config, obs, n_actions, n_agents, agent_id, device):
        self.config = config
        self.device = device
        self.agent_id = agent_id

        # ppo hyperparams - use config if available otherwise use defaults
        self.gamma = getattr(config, 'gamma', 0.99)
        self.gae_lambda = getattr(config, 'gae_lambda', 0.95)
        self.clip_coef = getattr(config, 'clip_coef', 0.2)
        self.ent_coef = getattr(config, 'ent_coef', 0.01)
        self.vf_coef = getattr(config, 'vf_coef', 0.5)
        self.max_grad_norm = getattr(config, 'max_grad_norm', 0.5)
        self.update_epochs = getattr(config, 'update_epochs', 4)
        self.num_minibatches = getattr(config, 'num_minibatches', 4)

        self.network = Agent(config, obs, n_actions, n_agents, agent_id, device).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.lr, eps=1e-5)

        # buffers to store experience during an episode
        self.obs_images = []
        self.obs_directions = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

        self.save_path = os.path.join(
            'models', getattr(config, 'model_name', 'ppo') + f'_agent_{agent_id}.pt'
        )

    def act(self, obs):
        with torch.no_grad():
            action, logprob, _, value = self.network.get_action_and_value(obs)
        return action.item(), logprob.item(), value.item()

    def store(self, obs, action, reward, done, logprob, value):
        self.obs_images.append(obs['image'])
        self.obs_directions.append(obs['direction'])
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def update(self, next_obs, next_done):
        num_steps = len(self.rewards)
        if num_steps == 0:
            return

        b_values = torch.tensor(self.values, dtype=torch.float32).to(self.device)
        b_rewards = torch.tensor(self.rewards, dtype=torch.float32).to(self.device)
        b_dones = torch.tensor(self.dones, dtype=torch.float32).to(self.device)
        b_logprobs = torch.tensor(self.logprobs, dtype=torch.float32).to(self.device)
        b_actions = torch.tensor(self.actions, dtype=torch.long).to(self.device)

        # compute GAE advantages
        with torch.no_grad():
            next_value = self.network.get_value(next_obs).reshape(-1).squeeze()
            advantages = torch.zeros(num_steps).to(self.device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - float(next_done)
                    nextval = next_value
                else:
                    nextnonterminal = 1.0 - b_dones[t + 1]
                    nextval = b_values[t + 1]
                delta = b_rewards[t] + self.gamma * nextval * nextnonterminal - b_values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + b_values

        minibatch_size = max(1, num_steps // self.num_minibatches)
        b_inds = np.arange(num_steps)

        for _ in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, num_steps, minibatch_size):
                mb_inds = b_inds[start:start + minibatch_size]

                mb_obs = {
                    'image': np.array(self.obs_images)[mb_inds],
                    'direction': np.array(self.obs_directions)[mb_inds],
                }

                _, newlogprob, entropy, newvalue = self.network.get_action_and_value(
                    mb_obs, b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

        # clear buffers for next episode
        self.obs_images = []
        self.obs_directions = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def save_model(self):
        torch.save(self.network.state_dict(), self.save_path)

    def load_model(self, save_path=None):
        path = save_path if save_path else self.save_path
        self.network.load_state_dict(torch.load(path, map_location=self.device))
