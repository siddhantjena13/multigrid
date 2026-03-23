from copy import deepcopy
import gym
from itertools import count
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torch
from torch.distributions.categorical import Categorical
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import wandb

from utils import plot_single_frame, make_video, extract_mode_from_path
from ppo_agent import PPOAgent

class MultiAgent():
    """This is a meta agent that creates and controls several sub agents. If model_others is True,
    Enables sharing of buffer experience data between agents to allow them to learn models of the 
    other agents. """

    def __init__(self, config, env, device, training=True, with_expert=None, debug=False):
        self.config = config
        self.env = env
        self.device = device
        self.training = training 
        self.with_expert = with_expert
        self.debug = debug 
        self.total_steps = 0
        self.agents = []
        self.n_agents = config.n_agents
        self.model_others = False
    
    def run_one_episode(self, env, episode, log=True, train=True, save_model=True, visualize=False):
        raise NotImplementedError("Subclasses must implement run_one_episode")

    def save_model_checkpoints(self, episode):
        if episode % self.config.save_model_episode == 0:
            for i in range(self.n_agents):
                self.agents[i].save_model()

    def print_terminal_output(self, episode, total_reward):
        if episode % self.config.print_every == 0:
            print('Total steps: {} \t Episode: {} \t Total reward: {}'.format(
                self.total_steps, episode, total_reward))

    def init_visualization_data(self, env, state):
        viz_data = {
            'agents_partial_images': [],
            'actions': [],
            'full_images': [],
            'predicted_actions': None
            }
        viz_data['full_images'].append(env.render('rgb_array'))

        if self.model_others:
            predicted_actions = []
            predicted_actions.append(self.get_action_predictions(state))
            viz_data['predicted_actions'] = predicted_actions

        return viz_data

    def add_visualization_data(self, viz_data, env, state, actions, next_state):
        viz_data['actions'].append(actions)
        viz_data['agents_partial_images'].append(
            [env.get_obs_render(
                self.get_agent_state(state, i)['image']) for i in range(self.n_agents)])
        viz_data['full_images'].append(env.render('rgb_array'))
        if self.model_others:
            viz_data['predicted_actions'].append(self.get_action_predictions(next_state))
        return viz_data
        
    def update_models(self):
        # Don't update model until you've taken enough steps in env
        if self.total_steps > self.config.initial_memory: 
            if self.total_steps % self.config.update_every == 0: # How often to update model
                raise NotImplementedError("Figure out how to actually update / train models!")
    
    def train(self, env):
        for episode in range(self.config.n_episodes):
            if episode % self.config.visualize_every == 0 and not (self.debug and episode == 0):
                viz_data = self.run_one_episode(env, episode, visualize=True)
                self.visualize(env, self.config.mode + '_training_step' + str(episode), 
                               viz_data=viz_data)
            else:
                self.run_one_episode(env, episode)

        env.close()
        return

    def visualize(self, env, mode, video_dir='videos', viz_data=None):
        if not viz_data:
            viz_data = self.run_one_episode(
                env, episode=0, log=False, train=False, save_model=False, visualize=True)
            env.close()

        video_path = os.path.join(*[video_dir, self.config.experiment_name, self.config.model_name])

        # Set up directory.
        if not os.path.exists(video_path):
            os.makedirs(video_path)

        # Get names of actions
        action_dict = {}
        for act in env.Actions:
            action_dict[act.value] = act.name

        traj_len = len(viz_data['rewards'])
        for t in range(traj_len):
            self.visualize_one_frame(t, viz_data, action_dict, video_path, self.config.model_name)
            print('Frame {}/{}'.format(t, traj_len))

        make_video(video_path, mode + '_trajectory_video')

    def visualize_one_frame(self, t, viz_data, action_dict, video_path, model_name):
        plot_single_frame(t,
                          viz_data['full_images'][t],
                          viz_data['agents_partial_images'][t],
                          viz_data['actions'][t],
                          viz_data['rewards'],
                          action_dict,
                          video_path,
                          self.config.model_name)

    def load_models(self, model_path=None):
        for i in range(self.n_agents):
            if model_path is not None:
                self.agents[i].load_model(save_path=model_path + '_agent_' + str(i))
            else:
                # Use agents' default model path
                self.agents[i].load_model()


class PPOMultiAgent(MultiAgent):
    # this is the actual implementation of MultiAgent using PPO
    # basically creates one PPOAgent per agent and coordinates them

    def __init__(self, config, env, device, training=True, with_expert=None, debug=False):
        super().__init__(config, env, device, training, with_expert, debug)

        # reset env once just to get the observation shape for building networks
        obs = env.reset()
        n_actions = config.n_actions

        for i in range(self.n_agents):
            agent_obs = self.get_agent_state(obs, i)
            agent = PPOAgent(config, agent_obs, n_actions, self.n_agents, i, device)
            self.agents.append(agent)

    def get_agent_state(self, state, agent_id):
        # image is per-agent (each sees its own partial view)
        # direction includes all agents so the network can condition on everyone's heading
        return {
            'image': state['image'][agent_id],
            'direction': state['direction'],
        }

    def run_one_episode(self, env, episode, log=True, train=True, save_model=True, visualize=False):
        self.env = env
        done = False
        rewards = []
        t = 0
        state = env.reset()

        if visualize:
            viz_data = self.init_visualization_data(env, state)

        while not done:
            self.total_steps += 1

            # get actions from all agents
            actions = []
            logprobs = []
            values = []
            for i in range(self.n_agents):
                obs = self.get_agent_state(state, i)
                action, logprob, value = self.agents[i].act(obs)
                actions.append(action)
                logprobs.append(logprob)
                values.append(value)

            # step the environment with all actions at once
            next_state, step_rewards, done, _ = env.step(actions)
            rewards.append(step_rewards)

            # store this timestep for each agent
            for i in range(self.n_agents):
                obs = self.get_agent_state(state, i)
                self.agents[i].store(obs, actions[i], step_rewards[i], done, logprobs[i], values[i])

            if visualize:
                viz_data = self.add_visualization_data(viz_data, env, state, actions, next_state)

            state = next_state
            t += 1

        # update each agent's network now that episode is done
        if train:
            for i in range(self.n_agents):
                next_obs = self.get_agent_state(state, i)
                self.agents[i].update(next_obs, done)

        if log:
            self.log_one_episode(episode, t, rewards)
        self.print_terminal_output(episode, np.sum(rewards))
        if save_model:
            self.save_model_checkpoints(episode)

        if visualize:
            viz_data['rewards'] = np.array(rewards)
            return viz_data

    def update_models(self):
        # ppo updates at end of each episode (handled in run_one_episode) so nothing here
        pass

    def log_one_episode(self, episode, t, rewards):
        rewards_arr = np.array(rewards)  # shape: (timesteps, n_agents)

        log_dict = {
            'episode/x_axis': episode,
            'episode/total_reward': float(np.sum(rewards_arr)),
            'episode/episode_length': t,
        }
        for i in range(self.n_agents):
            log_dict[f'episode/agent_{i}_reward'] = float(np.sum(rewards_arr[:, i]))

        wandb.log(log_dict)
