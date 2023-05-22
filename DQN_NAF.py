import gym
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class DQN_NAF:
    def __init__(self, state_dim, action_dim, action_bounds):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99
        self.tau = 0.001
        self.q_learning_rate = 0.0001
        self.actor_learning_rate = 0.0001
        self.critic_learning_rate = 0.001
        self.q_network = self._build_q_network()
        self.target_network = self._build_q_network()
        self.actor_network = self._build_actor_network()
        self.critic_network = self._build_critic_network()

    def _build_q_network(self):
        input_state = Input(shape=(self.state_dim,))
        input_action = Input(shape=(self.action_dim,))
        h1 = Dense(64, activation='relu')(input_state)
        h2 = Dense(64, activation='relu')(h1)
        h3 = Dense(64, activation='relu')(h2)
        q_values = Dense(1, activation='linear')(h3)
        mu = Dense(self.action_dim, activation='tanh')(h3)
        sigma = Dense(self.action_dim, activation='softplus')(h3)
        norm_dist = Lambda(lambda x: x * self.action_bounds)(mu)
        model = Model(inputs=[input_state, input_action], outputs=[q_values, norm_dist, sigma])
        model.compile(optimizer=Adam(lr=self.q_learning_rate), loss='mse')
        return model

    def _build_actor_network(self):
        input_state = Input(shape=(self.state_dim,))
        h1 = Dense(64, activation='relu')(input_state)
        h2 = Dense(64, activation='relu')(h1)
        h3 = Dense(64, activation='relu')(h2)
        mu = Dense(self.action_dim, activation='tanh')(h3)
        sigma = Dense(self.action_dim, activation='softplus')(h3)
        norm_dist = Lambda(lambda x: x * self.action_bounds)(mu)
        model = Model(inputs=input_state, outputs=[norm_dist, sigma])
        model.compile(optimizer=Adam(lr=self.actor_learning_rate), loss='mse')
        return model

    def _build_critic_network(self):
        input_state = Input(shape=(self.state_dim,))
        input_action = Input(shape=(self.action_dim,))
        h1 = Dense(64, activation='relu')(input_state)
        h2 = Dense(64, activation='relu')(h1)
        h3 = Dense(64, activation='relu')(h2)
        h4 = Dense(64, activation='relu')(input_action)
        h5 = Dense(64, activation='relu')(h3)
        h6 = Dense(64, activation='relu')(h4)
        h7 = tf.keras.layers.Concatenate()([h5, h6])
        state_value = Dense(1, activation='linear')(h7)
        model = Model(inputs=[input_state, input_action], outputs=state_value)
        model.compile(optimizer=Adam(lr=self.critic_learning_rate), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        action, _ = self.actor_network.predict(np.expand_dims(state, axis=0))
        return action[0]

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.asarray(states)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        next_states = np.asarray(next_states)
        dones = np.asarray(dones)

        # Compute target Q values
        _, next_actions, next_sigmas = self.actor_network.predict(next_states)
        target_q_values = self.target_network.predict([next_states, next_actions])[0]
        next_values = target_q_values - next_sigmas**2 / 2
        target_q_values = rewards + (1 - dones) * self.gamma * next_values

        # Train Q network
        self.q_network.train_on_batch([states, actions], [target_q_values, actions, actions])

        # Train actor and critic networks
        with tf.GradientTape() as tape:
            mu, sigma = self.actor_network(states)
            values = self.critic_network([states, mu])
            actor_loss = -tf.reduce_mean(values)
        actor_gradients = tape.gradient(actor_loss, self.actor_network.trainable_variables)
        self.actor_network.optimizer.apply_gradients(zip(actor_gradients, self.actor_network.trainable_variables))

        with tf.GradientTape() as tape:
            target_actions, target_sigmas = self.actor_network(next_states)
            target_values = self.target_network([next_states, target_actions])
            critic_target = rewards + (1 - dones) * self.gamma * target_values
            critic_loss = tf.reduce_mean((critic_target - values)**2)
        critic_gradients = tape.gradient(critic_loss, self.critic_network.trainable_variables)
        self.container_network.optimizer.apply_gradients(zip(critic_gradients, self.critic_network.trainable_variables))

    # step 11
    def update_target_network(self):
        q_weights = self.q_network.get_weights()
        target_weights = self.target_network.get_weights()
        for i in range(len(q_weights)):
            target_weights[i] = self.tau * q_weights[i] + (1 - self.tau) * target_weights[i]
        self.target_network.set_weights(target_weights)

    def train(self, episodes):
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                self.replay()
                self.update_target_network()
            print("Episode:", episode, "Total reward:", total_reward)


# Set hyperparameters
gamma = 0.99
tau = 0.001
q_learning_rate = 0.0001
actor_learning_rate = 0.0001
critic_learning_rate = 0.001
buffer_size = 100000
batch_size = 64
action_bounds = 1

# Set up the environment
env = gym.make('RoboticArm-v0')
state_dim = env.observation_space.shape[0]
action_dim= env.action_space.shape[0]
action_bounds = env.action_space.high
# Create DQN_NAF agent and train
agent = DQN_NAF(state_dim, action_dim, action_bounds)
agent.train(episodes=1000)