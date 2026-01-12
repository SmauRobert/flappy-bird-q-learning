import random
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .model import Model


class FlappyAgent:
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_actions: int,
        device: torch.device,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        buffer_size: int = 50000,
        initial_weights: Optional[str] = None,
    ):
        self.device = device
        self.gamma = gamma
        self.num_actions = num_actions

        # Policy Net: The active network we train
        self.policy_net = Model(input_shape, num_actions).to(device)

        # Target Net: The stable network for calculating future rewards
        self.target_net = Model(input_shape, num_actions).to(device)

        # --- Load Initial Weights if provided ---
        if initial_weights:
            print(f"Loading initial weights from: {initial_weights}")
            try:
                # Load weights (map_location ensures it works across CPU/GPU)
                weights = torch.load(initial_weights, map_location=device)
                self.policy_net.load_state_dict(weights)
                print("✅ Weights loaded successfully.")
            except Exception as e:
                print(f"❌ Error loading weights: {e}")

        # Copy weights to target net (whether random or loaded)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # --- 2. The Optimizer ---
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # --- 3. The Memory ---
        self.memory: Deque = deque(maxlen=buffer_size)

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        Epsilon-Greedy Action Selection with Biased Exploration.
        """
        # 1. Exploration (Random)
        if random.random() < epsilon:
            return np.random.choice([0, 1], p=[0.9, 0.1])

        # 2. Exploitation (Model)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                return q_values.argmax().item()

    def cache(self, state, action, reward, next_state, done):
        """Stores experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def recall(self, batch_size: int):
        """Samples a random batch of experiences."""
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state),
            np.array(done, dtype=np.uint8),
        )

    def learn(self, batch_size: int = 32) -> float:
        """Updates the Policy Network using a batch of experiences."""
        if len(self.memory) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.recall(batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        current_q = self.policy_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0].unsqueeze(1)
            expected_q = rewards_t + (self.gamma * next_q * (1 - dones_t))

        loss = nn.SmoothL1Loss()(current_q, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def sync_target(self):
        """Copies weights from Policy Net to Target Net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
