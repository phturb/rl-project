

import gym
import os
import json
import copy

import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
from functools import partial

from keras.models import Sequential, clone_model, load_model
from keras.layers import Dense, Lambda, Embedding, Reshape
from tensorflow.keras.optimizers import Adam
from collections import deque


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
        """
        Return the action for the given state from a model.
        """

        if not warmup:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.espilon_decay)
        if warmup or (training and (np.random.rand() <= self.epsilon)):
            return self.env.action_space.sample()
        else:
            return np.argmax(model.predict(state.reshape(1, -1)))

class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)
        self.maxlen = max_size
        self.len = 0

    def add(self, experience):
        """
        Add experience to the memory.
        """
        self.len = min(self.len + 1, self.maxlen)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Return a random sample of the memory of the size of batch_size.
        """
        indexes = np.random.choice(np.arange(self.len), size=batch_size, replace=False)
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
        dones = 1 - np.array(dones)

        # Calculate the target
        targets = self.model.predict(states)
        if self.ddqn:
            Q_sa = np.max(self.target_model.predict(next_states), axis=1)
        else:
            Q_sa = np.max(self.model.predict(next_states), axis=1)
        adjusted_rewards = rewards + dones * gamma * Q_sa
        for i in range(batch_size):
            targets[i][actions[i]] = adjusted_rewards[i]

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
        episodes_rewards = []
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
            episodes_rewards.append(episode_reward)
            print("Total Steps: {}\t\tEpisode: {}\t\tReward: {},\t\tEpsilon: {},\t\tSteps: {}".format(n_steps, n_episodes ,episode_reward, self.policy.epsilon ,steps))
        episodes_rewards.pop()
        return history, episodes_rewards
        
    def test(self, n_tests=1, success_average=1, render=False):
        """
        Test the model
        """
        total_reward_array = np.zeros(n_tests)
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
            total_reward_array[test] = episode_reward
            print("Test {}\t\tReward: {},\t\tSteps: {}".format(test + 1 ,episode_reward, steps))
        average = float(total_reward_array.sum())/float(n_tests)
        print("Average Success: {} - Experience {}".format(average, {True: "Success", False: "Failed"}[average >= success_average]))    


    def save(self, path):
        """
        Save the online model
        """
        self.model.save(path)

    def load(self, path):
        """
        Load the model and compile both models
        """
        self.model = load_model(path)
        self.compile()


def plot_rewards(rewards, name, path):
    """
    Plot the rewards
    """
    plt.plot(rewards)
    plt.title(name)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(path)
    plt.close()

def plot_compare_rewards(rewards_list,labels_list, name, path):
    """
    Plot the comparaison rewards
    """
    for rewards in rewards_list:
        plt.plot(rewards)
    plt.legend(labels_list)
    plt.title(name)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(path)
    plt.close()

def run_from_config(config, render_tests=False):
    """
    Run the agent from a config file.
    """

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
    if config['model_path'] is not None and os.path.exists(config['model_path']):
        print("Model is already trained, loading the weights from {}".format(config['model_path']))
        agent.load(config['model_path'])
        policy.epsilon = policy.epsilon_min
        history = []
        with open(config['rewards_path'], "r") as f:
            train_rewards = json.load(f)
    else:
        agent.compile()
        history, train_rewards = agent.train(max_steps=train_config['max_steps'], batch_size=train_config['batch_size'], gamma=train_config['gamma'])
        agent.save(config['model_path'])
    agent.test(n_tests=test_config['n_tests'], render=render_tests)
    with open(config['rewards_path'], "w") as f:
        json.dump(train_rewards, f)
    return history, train_rewards, agent

def render_from_config(config):
    """
    Run the agent from a config file.
    """
    print("Running from config: {}".format(config['env']))
    memory_config = config['memory_config']
    policy_config = config['policy_config']
    agent_config = config['agent_config']
    env = create_gym_env(config['env'], config['seed'])
    model = create_model(config['input_shape'] , env.action_space.n, layers=config['layers'], dueling=config['dueling'], embedding=config['embedding'])
    memory = Memory(max_size=memory_config['max_size'])
    policy = PolicyDiscreet(env, policy_config["epsilon"], policy_config['epsilon_decay'], policy_config['epsilon_min'])
    policy.epsilon = policy.epsilon_min
    agent = Agent(model, env, memory, warmup_steps=agent_config['warmup_steps'], target_model_update=agent_config['target_model_update'], policy=policy, ddqn=True)
    if config['model_path'] is not None and os.path.exists(config['model_path']):
        print("Model is already trained, loading the weights from {}".format(config['model_path']))
        agent.load(config['model_path'])
    else:
        raise Exception("Model not found")
    agent.test(n_tests=1, render=True)
    return