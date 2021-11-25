

import gym
import os
import numpy as np

import keras.backend as K
# import tensorflow as tf

from keras.models import Sequential, clone_model, load_model
from keras.layers import Dense, Lambda, Embedding, Reshape
from tensorflow.keras.optimizers import Adam
from collections import deque


ENV_NAME = ['Taxi-v3', 'CartPole-v1', 'Blackjack-v1', 'FrozenLake-v1', 'MountainCar-v0']


def create_gym_env(env_name, seed):
    """
    Create a gym environment.
    """
    env = gym.make(env_name)
    env.seed(seed)
    np.random.seed(seed)
    return env

def create_model(input_shape, output_shape, layers=[32, 64], dueling=False, embedding=None):
    """
    Create a model for the self-play game.
    """
    if len(layers) < 1:
        raise ValueError("At least one layer is required.")

    model = Sequential(name='DQN')
    if embedding is not None:
        model.add(Embedding(embedding[0], embedding[1], input_shape=input_shape, name='Input'))
        model.add(Reshape(embedding[2]))
        model.add(Dense(layers[0], activation="relu", name='layer_0'))
    else:
        model.add(Dense(layers[0], input_shape=input_shape, activation="relu", name='Input'))
    for i, layer in enumerate(layers[1:]):
        model.add(Dense(layer, activation="relu", name='layer_' + str(i + 1)))
    if dueling:
        model.add(Dense(output_shape + 1, activation='linear', name="pre_actions"))
        model.add(Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], axis=1, keepdims=True), output_shape=(output_shape,), name="actions"))
    else:
        model.add(Dense(output_shape, activation='linear', name="actions"))
    return model

class PolicyDiscreet:
    """
    Epsilon greedy policy for a deiscrete action
    """
    def __init__(self, env, epsilon, espilon_decay, epsilon_min):
        self.env = env
        self.epsilon = epsilon
        self.espilon_decay = espilon_decay
        self.epsilon_min = epsilon_min

    def __call__(self, model, state, training=False, warmup=False):

        if not warmup:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.espilon_decay)
        if warmup or (training and (np.random.rand() <= self.epsilon)):
            return self.env.action_space.sample()
        else:
            return np.argmax(model.predict(state.reshape(1, -1)))

class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[i] for i in indexes]

class Agent:
    """
    RL Agent to run the open ai gym environment.
    """

    def __init__(self, model, env, memory, warmup_steps, target_model_update, policy, ddqn=True):
        self.model = model
        self.env = env
        self.memory = memory
        self.warmup_steps = warmup_steps
        self.target_model_update = target_model_update
        self.policy = policy
        self.ddqn = ddqn

    def compile(self):
        """
        Compile the model.
        """
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
        self.target_model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
        self.model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
        self.model.summary()

    def update_target_model_hard(self):
        """
        Update the target model weight to match the online model
        """
        self.target_model.set_weights(self.model.get_weights())

    def backward(self, batch_size, gamma, n_steps):
        """
        Backward pass.
        """
        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        # Calculate the target
        targets = self.model.predict(states)
        if self.ddqn:
            Q_sa = self.target_model.predict(next_states)
        else:
            Q_sa = self.model.predict(next_states)
        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + gamma * np.max(Q_sa[i])

        hist = self.model.fit(states, targets, epochs=1, verbose=0)

        if self.ddqn and n_steps % self.target_model_update == 0:
            self.update_target_model_hard()

        return hist

    def train(self, max_steps=10000, batch_size=32, gamma=0.95, render=False):
        """
        Train the model.
        """
        n_steps = 0
        n_episodes = 0
        history = []
        while(n_steps <= max_steps + self.warmup_steps):
            n_episodes += 1
            episode_reward = 0
            steps = 0
            state = np.array(self.env.reset())
            done = False
            if render:
                self.env.render()
            while(not done):
                n_steps += 1
                steps += 1
                action = self.policy(self.model, state, training=True, warmup=(n_steps <= self.warmup_steps))
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.array(next_state)
                episode_reward += reward
                # Add to memory the experience
                self.memory.add((state, action, reward, next_state, done))

                if self.warmup_steps < n_steps:
                    history.append(self.backward(batch_size, gamma, n_steps))

                state = next_state
                if render:
                    self.env.render()
                if n_steps > max_steps + self.warmup_steps:
                    break

            print("Total Steps: {}\t\tEpisode: {}\t\tReward: {},\t\tEpsilon: {},\t\tSteps: {}".format(n_steps, n_episodes ,episode_reward, self.policy.epsilon ,steps))
        return history
        
    def test(self, n_tests=1, success_average=1, render=False):
        """
        Test the model
        """
        total_reward = 0
        for test in range(n_tests):
            steps = 0
            episode_reward = 0
            state = np.array(self.env.reset())
            done = False
            if render:
                self.env.render()
            while(not done):
                steps+=1
                action = self.policy(self.model, state, training=False, warmup=False)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.array(next_state)
                episode_reward += reward
                state = next_state
                if render:
                    self.env.render()
            total_reward += episode_reward
            print("Test {}\t\tReward: {},\t\tSteps: {}".format(test + 1 ,episode_reward, steps))
        average = float(total_reward)/float(n_tests)
        print("Average Success: {} - Experience {}".format(average, {True: "Success", False: "Failed"}[average >= success_average]))    


    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = load_model(path)
        self.compile()

def run_from_config(config, render_tests=False):
    print("Running from config: {}".format(config['env']))
    memory_config = config['memory_config']
    policy_config = config['policy_config']
    agent_config = config['agent_config']
    train_config = config['train_config']
    test_config = config['test_config']
    env = create_gym_env(config['env'], config['seed'])
    model = create_model(config['input_shape'] , env.action_space.n, layers=config['layers'], dueling=config['dueling'], embedding=config['embedding'])
    memory = Memory(max_size=memory_config['max_size'])
    policy = PolicyDiscreet(env, policy_config["epsilon"], policy_config['epsilon_decay'], policy_config['epsilon_min'])
    agent = Agent(model, env, memory, warmup_steps=agent_config['warmup_steps'], target_model_update=agent_config['target_model_update'], policy=policy, ddqn=True)
    if config['load_path'] is not None and os.path.exists(config['load_path']):
        agent.load(config['load_path'])
        policy.epsilon = policy.epsilon_min
        history = []
    else:
        agent.compile()
        history = agent.train(max_steps=train_config['max_steps'], batch_size=train_config['batch_size'], gamma=train_config['gamma'])
    agent.test(n_tests=test_config['n_tests'], render=render_tests)
    agent.save(config['load_path'])
    return history, agent