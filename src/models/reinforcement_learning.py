"""
High-Frequency Trading Analytics System
Reinforcement Learning Models for Trading

This module implements reinforcement learning models for high-frequency trading,
including DQN, PPO, and A3C algorithms for market making and execution optimization.

Author: Gabriel Demetrios Lafis
Date: June 2025
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from collections import deque, namedtuple
import random
import gym
from gym import spaces
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define experience replay memory
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 100000):
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum size of the buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """Deep Q-Network for market making."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """Initialize DQN network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Size of hidden layers
        """
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state):
        """Forward pass through network."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_values = self.fc4(x)
        return q_values


class DQNAgent:
    """DQN agent for market making."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Size of hidden layers
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Decay rate for exploration
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
            device: Device to run the model on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        self.steps = 0
        
        # Initialize networks
        self.policy_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
    
    def select_action(self, state, training: bool = True):
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randrange(self.action_dim)
        else:
            # Exploitation: best action according to policy
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def update_epsilon(self):
        """Update exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def update(self):
        """Update policy network from replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute V(s_{t+1}) for all next states
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        
        # Compute the expected Q values
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(q_values, expected_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Update exploration rate
        self.update_epsilon()
        
        return loss.item()
    
    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model weights."""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.steps = checkpoint['steps']
            logger.info(f"Model loaded from {path}")
        else:
            logger.error(f"No model found at {path}")


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO algorithm."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, continuous: bool = False):
        """Initialize Actor-Critic network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Size of hidden layers
            continuous: Whether action space is continuous
        """
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy) network
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Critic (value) network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.continuous = continuous
        
        if continuous:
            # For continuous action spaces (e.g., spread adjustment)
            self.action_mean = nn.Linear(hidden_dim, action_dim)
            self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))
        else:
            # For discrete action spaces (e.g., buy/sell/hold)
            self.action_probs = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state):
        """Forward pass through network."""
        features = self.feature_extractor(state)
        
        # Actor output
        actor_features = self.actor(features)
        if self.continuous:
            action_mean = self.action_mean(actor_features)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            return action_mean, action_std, self.critic(features)
        else:
            action_probs = F.softmax(self.action_probs(actor_features), dim=-1)
            return action_probs, self.critic(features)


class PPOAgent:
    """PPO agent for market making and execution optimization."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_param: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        continuous: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize PPO agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Size of hidden layers
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_param: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy coefficient
            max_grad_norm: Maximum gradient norm
            continuous: Whether action space is continuous
            device: Device to run the model on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.continuous = continuous
        self.device = device
        
        # Initialize actor-critic network
        self.ac_network = ActorCritic(state_dim, action_dim, hidden_dim, continuous).to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.ac_network.parameters(), lr=learning_rate)
        
        # Initialize memory for trajectories
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def select_action(self, state, training: bool = True):
        """Select action using current policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if self.continuous:
                action_mean, action_std, value = self.ac_network(state_tensor)
                
                if training:
                    # Sample from normal distribution
                    dist = Normal(action_mean, action_std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(dim=-1)
                else:
                    # Use mean action for evaluation
                    action = action_mean
                    dist = Normal(action_mean, action_std)
                    log_prob = dist.log_prob(action).sum(dim=-1)
                
                action = action.cpu().numpy()[0]
            else:
                action_probs, value = self.ac_network(state_tensor)
                
                if training:
                    # Sample from categorical distribution
                    dist = Categorical(action_probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                else:
                    # Use most probable action for evaluation
                    action = torch.argmax(action_probs, dim=-1)
                    dist = Categorical(action_probs)
                    log_prob = dist.log_prob(action)
                
                action = action.item()
            
            value = value.item()
            log_prob = log_prob.item()
        
        if training:
            self.states.append(state)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.values.append(value)
        
        return action
    
    def store_transition(self, reward, done):
        """Store reward and done flag for current transition."""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def update(self, next_state=None, next_value=None, epochs: int = 10):
        """Update policy using collected trajectories."""
        # If trajectory is not complete, estimate final value
        if next_state is not None and next_value is None:
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                if self.continuous:
                    _, _, next_value = self.ac_network(next_state_tensor)
                else:
                    _, next_value = self.ac_network(next_state_tensor)
                next_value = next_value.item()
        
        # Convert lists to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        if self.continuous:
            actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        else:
            actions = torch.LongTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        # Compute returns and advantages
        returns, advantages = self._compute_gae(rewards, values, dones, next_value)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(epochs):
            if self.continuous:
                action_mean, action_std, values_pred = self.ac_network(states)
                dist = Normal(action_mean, action_std)
                new_log_probs = dist.log_prob(actions).sum(dim=-1)
                entropy = dist.entropy().mean()
            else:
                action_probs, values_pred = self.ac_network(states)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
            
            values_pred = values_pred.squeeze(-1)
            
            # Compute ratio and clipped ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            
            # Compute losses
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values_pred, returns)
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac_network.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        # Clear memory
        self._clear_memory()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': loss.item()
        }
    
    def _compute_gae(self, rewards, values, dones, next_value=0):
        """Compute returns and advantages using Generalized Advantage Estimation."""
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * next_non_terminal - values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
            returns[t] = advantages[t] + values[t]
        
        return returns, advantages
    
    def _clear_memory(self):
        """Clear trajectory memory."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'ac_network': self.ac_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model weights."""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.ac_network.load_state_dict(checkpoint['ac_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info(f"Model loaded from {path}")
        else:
            logger.error(f"No model found at {path}")


class MarketMakingEnv:
    """Market making environment for reinforcement learning."""
    
    def __init__(
        self,
        data_path: str,
        window_size: int = 10,
        max_inventory: int = 100,
        transaction_cost: float = 0.0002,
        reward_scaling: float = 1.0,
        risk_aversion: float = 0.1
    ):
        """Initialize market making environment.
        
        Args:
            data_path: Path to market data
            window_size: Size of observation window
            max_inventory: Maximum allowed inventory
            transaction_cost: Cost per transaction
            reward_scaling: Scaling factor for rewards
            risk_aversion: Risk aversion parameter
        """
        # Load market data
        self.data = pd.read_parquet(data_path)
        self.window_size = window_size
        self.max_inventory = max_inventory
        self.transaction_cost = transaction_cost
        self.reward_scaling = reward_scaling
        self.risk_aversion = risk_aversion
        
        # Define action and observation spaces
        # Actions: 0=sell, 1=hold, 2=buy
        self.action_space = spaces.Discrete(3)
        
        # Observation: price features + inventory
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size * 5 + 1,),  # OHLCV * window_size + inventory
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self):
        """Reset environment to initial state."""
        self.current_step = self.window_size
        self.inventory = 0
        self.cash = 0
        self.done = False
        self.trades = []
        
        return self._get_observation()
    
    def step(self, action):
        """Take action in environment."""
        if self.done:
            return self._get_observation(), 0, self.done, {}
        
        # Get current price data
        current_price = self.data.iloc[self.current_step]
        mid_price = (current_price['ask'] + current_price['bid']) / 2
        
        # Execute action
        reward = 0
        if action == 0:  # Sell
            if self.inventory > 0:
                # Sell at bid price
                sell_price = current_price['bid']
                reward += (sell_price - self.transaction_cost * sell_price) * 1
                self.cash += sell_price
                self.inventory -= 1
                self.trades.append(('sell', self.current_step, sell_price))
            else:
                # Penalty for invalid action
                reward -= 0.1
        
        elif action == 2:  # Buy
            if self.inventory < self.max_inventory:
                # Buy at ask price
                buy_price = current_price['ask']
                reward -= (buy_price + self.transaction_cost * buy_price) * 1
                self.cash -= buy_price
                self.inventory += 1
                self.trades.append(('buy', self.current_step, buy_price))
            else:
                # Penalty for invalid action
                reward -= 0.1
        
        # Add inventory risk penalty
        inventory_risk = self.risk_aversion * (self.inventory ** 2)
        reward -= inventory_risk
        
        # Add mark-to-market P&L
        if self.inventory != 0:
            reward += self.inventory * (current_price['mid'] - self.data.iloc[self.current_step - 1]['mid'])
        
        # Scale reward
        reward *= self.reward_scaling
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        if self.current_step >= len(self.data) - 1:
            self.done = True
            
            # Add final liquidation reward/penalty
            if self.inventory != 0:
                final_price = self.data.iloc[self.current_step]['bid'] if self.inventory > 0 else self.data.iloc[self.current_step]['ask']
                liquidation_value = self.inventory * final_price
                liquidation_cost = abs(self.inventory) * final_price * self.transaction_cost
                reward += liquidation_value - liquidation_cost
        
        return self._get_observation(), reward, self.done, {}
    
    def _get_observation(self):
        """Get current observation."""
        # Extract window of OHLCV data
        obs_window = self.data.iloc[self.current_step - self.window_size:self.current_step]
        
        # Normalize features
        price_mean = obs_window['mid'].mean()
        price_std = obs_window['mid'].std() + 1e-6
        
        # Extract and normalize features
        bid = (obs_window['bid'].values - price_mean) / price_std
        ask = (obs_window['ask'].values - price_mean) / price_std
        mid = (obs_window['mid'].values - price_mean) / price_std
        volume = obs_window['volume'].values / obs_window['volume'].max()
        spread = (obs_window['ask'] - obs_window['bid']).values / price_std
        
        # Combine features
        features = np.concatenate([
            bid, ask, mid, volume, spread,
            [self.inventory / self.max_inventory]  # Normalized inventory
        ])
        
        return features
    
    def render(self):
        """Render environment state."""
        current_price = self.data.iloc[self.current_step]
        print(f"Step: {self.current_step}, Bid: {current_price['bid']:.2f}, Ask: {current_price['ask']:.2f}, Inventory: {self.inventory}, Cash: {self.cash:.2f}")


# Example usage
if __name__ == "__main__":
    # Create market making environment
    env = MarketMakingEnv(
        data_path="data/historical/btcusdt_level1_202506.parquet",
        window_size=10,
        max_inventory=10,
        transaction_cost=0.0002,
        reward_scaling=1.0,
        risk_aversion=0.1
    )
    
    # Initialize PPO agent
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=128,
        learning_rate=3e-4,
        gamma=0.99,
        continuous=False
    )
    
    # Training loop
    num_episodes = 100
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            agent.store_transition(reward, done)
            
            # Update state
            state = next_state
            episode_reward += reward
        
        # Update agent
        update_info = agent.update()
        
        # Log progress
        logger.info(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}, Loss: {update_info['total_loss']:.4f}")
        
        # Save model periodically
        if (episode + 1) % 10 == 0:
            agent.save(f"models/ppo_market_making_ep{episode+1}.pt")

