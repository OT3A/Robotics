import tensorflow as tf
import numpy as np
import random
from collections import deque
from math import sqrt
from typing import List, Tuple, Callable, Any, Union

class DQNAF:
    def __init__(self, state_size: int, action_size: int, hidden_size: int, memory_size: int, batch_size: int, gamma: float, tau: float, sigma: float, mu: float, theta: float, policy: Callable[[tf.Tensor], tf.Tensor], q_function: Callable[[tf.Tensor], tf.Tensor], optimizer: tf.keras.optimizers.Optimizer):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.sigma = sigma
        self.mu = mu
        self.theta = theta
        self.policy = policy
        self.q_function = q_function
        self.optimizer = optimizer

        self.memory = deque(maxlen=self.memory_size)
        self.mu_vector = np.ones(self.action_size) * self.mu
        self.theta_matrix = np.identity(self.action_size) * self.theta
        self.covariance_matrix = np.identity(self.action_size) * self.sigma

        self.target_mu_vector = np.zeros(self.action_size)
        self.target_theta_matrix = np.zeros((self.action_size, self.action_size))
        self.target_covariance_matrix = np.zeros((self.action_size, self.action_size))
        self.memory_index = 0
        self.training_step = 0

    def train(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))
        self.memory_index = (self.memory_index + 1) % self.memory_size

        if len(self.memory) >= self.batch_size:
            self.training_step += 1

            # Sample a batch of transitions from memory
            batch = self.sample_batch()

            # Compute target Q-values for the batch
            target_q_values = self.compute_target_q_values(batch)

            # Update the Q-function with the batch
            self.update_q_function(batch, target_q_values)

            # Update the policy with the batch
            self.update_policy(batch)

            # Update the target networks
            self.update_target_networks()

    def sample_batch(self) -> List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]:
        indices = random.sample(range(len(self.memory)), self.batch_size)
        batch = [self.memory[i] for i in indices]
        return batch

    def compute_target_q_values(self, batch: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]) -> np.ndarray:
        target_q_values = np.zeros(len(batch))

        for i, (state, action, reward, next_state, done) in enumerate(batch):
            next_action = self.sample_action(next_state)

            if done:
                target_q_values[i] = reward
            else:
                next_q_value = self.compute_next_q_value(next_state, next_action)
                target_q_values[i] = reward + self.gamma * next_q_value

        return target_q_values

    @tf.function
    def compute_q_values(self, state: tf.Tensor, action: tf.Tensor) -> tf.Tensor:
        state_action = tf.concat([state, action], axis=-1)
        hidden = tf.keras.layers.Dense(self.hidden_size, activation='relu')(state_action)
        q_values = tf.keras.layers.Dense(self.action_size)(hidden)
        return q_values

    def update_q_function(self, batch: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]], target_q_values: np.ndarray) -> None:
        states = np.array([transition[0] for transition in batch])
        actions = np.array([transition[1] for transition in batch])
        targets = target_q_values.reshape(-1, 1)

        with tf.GradientTape() as tape:
            q_values = self.compute_q_values(tf.convert_to_tensor(states, dtype=tf.float32), tf.convert_to_tensor(actions, dtype=tf.float32))
            loss = tf.keras.losses.MSE(targets, q_values)

        gradients = tape.gradient(loss, self.q_function.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_function.trainable_variables))

    def update_policy(self, batch: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]) -> None:
        states = np.array([transition[0] for transition in batch])
        actions = np.array([transition[1] for transition in batch])
        rewards = np.array([transition[2] for transition in batch])
        next_states = np.array([transition[3] for transition in batch])
        dones = np.array([transition[4] for transition in batch])

        # Compute advantages for the batch
        advantages = self.compute_advantages(states, actions, rewards, next_states, dones)

        # Normalize the advantages
        mean_advantage = np.mean(advantages)
        std_advantage = np.std(advantages)
        normalized_advantages = (advantages - mean_advantage) / (std_advantage + 1e-8)

        with tf.GradientTape() as tape:
            mu_vector, theta_matrix, covariance_matrix = self.compute_policy(tf.convert_to_tensor(states, dtype=tf.float32))
            policy_loss = self.compute_policy_loss(mu_vector, theta_matrix, covariance_matrix, tf.convert_to_tensor(actions, dtype=tf.float32), tf.convert_to_tensor(normalized_advantages, dtype=tf.float32))

        gradients = tape.gradient(policy_loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))

    def update_target_networks(self) -> None:
        for source, target in zip(self.q_function.trainable_variables, self.target_q_function.trainable_variables):
            target.assign(self.tau * source + (1 - self.tau) * target)

        self.target_mu_vector = self.tau * self.mu_vector + (1 - self.tau) * self.target_mu_vector
        self.target_theta_matrix = self.tau * self.theta_matrix + (1 - self.tau) * self.target_theta_matrix
        self.target_covariance_matrix = self.tau * self.covariance_matrix + (1 - self.tau) * self.target_covariance_matrix

    def compute_advantages(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray) -> np.ndarray:
        # Compute target Q-values for the next states
        next_actions = self.sample_actions(next_states)
        next_q_values = self.compute_q_values(tf.convert_to_tensor(next_states, dtype=tf.float32), tf.convert_to_tensor(next_actions, dtype=tf.float32))
        next_q_values = next_q_values.numpy().reshape(-1)

        # Compute next state value for the next states
        next_v_values = self.compute_next_v_values(next_states, next_actions)

        # Compute advantages for the batch
        advantages = rewards + (1 - dones) * self.gamma * next_v_values - self.compute_q_values(tf.convert_to_tensor(states, dtype=tf.float32), tf.convert_to_tensor(actions, dtype=tf.float32)).numpy().reshape(-1) + next_q_values - next_v_values

        return advantages

    def compute_policy(self, state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # Compute policy parameters for the state
        hidden = tf.keras.layers.Dense(self.hidden_size, activation='relu')(state)
        mu_vector = tf.keras.layers.Dense(self.action_size)(hidden)
        mu_vector = tf.keras.activations.tanh(mu_vector) * self.mu_vector
        theta_matrix = tf.keras.layers.Dense(self.action_size * self.action_size)(hidden)
        theta_matrix = tf.reshape(theta_matrix, (-1, self.action_size, self.action_size))
        theta_matrix = tf.linalg.band_part(theta_matrix, -1, 0)
        theta_matrix = theta_matrix + tf.transpose(theta_matrix, perm=[0, 2, 1]) - tf.linalg.diag_part(theta_matrix)
        theta_matrix = theta_matrix * self.theta_matrix
        covariance_matrix = tf.linalg.expm(theta_matrix)
        covariance_matrix = tf.matmul(covariance_matrix, tf.transpose(covariance_matrix, perm=[0, 2, 1]))
        covariance_matrix = covariance_matrix * self.covariance_matrix
        return mu_vector, theta_matrix, covariance_matrix

    @tf.function
    def sample_action(self, state: tf.Tensor) -> tf.Tensor:
        mu_vector, theta_matrix, covariance_matrix = self.compute_policy(state)
        action = tf.random.normal((self.action_size,), mean=mu_vector, stddev=tf.sqrt(tf.linalg.diag_part(covariance_matrix)))
        return action

    @tf.function
    def sample_actions(self, states: tf.Tensor) -> tf.Tensor:
        mu_vectors, theta_matrices, covariance_matrices = self.compute_policy(states)
        actions = tf.random.normal((len(states), self.action_size), mean=mu_vectors, stddev=tf.sqrt(tf.linalg.diag_part(covariance_matrices)))
        return actions

    def compute_next_q_value(self, next_state: np.ndarray, next_action: np.ndarray) -> float: