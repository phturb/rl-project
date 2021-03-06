

from genericpath import exists
import gym
import os
import json
import copy
import itertools

import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from functools import partial
from datetime import datetime

from keras.models import Sequential, clone_model, load_model
from keras.layers import Dense, Lambda, Embedding, Reshape, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque
from minatar import Environment

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)): 
            return None

        return json.JSONEncoder.default(self, obj)


def create_gym_env(env_name, seed):
    """
    Create a gym environment.
    """
    if env_name.startswith("minatar"):
        env_name = env_name.replace("minatar-", "")
        env = Environment(env_name)
    else:
        env = gym.make(env_name)
        env.seed(seed)
    np.random.seed(seed)
    return env


def create_model(input_shape, output_shape, layers=[32, 64], dueling=False, embedding=None, minatar=False, duelling=False):
    """
    Create a model for the self-play game.
    """
    if len(layers) < 1:
        raise ValueError("At least one layer is required.")

    model = Sequential(name={True: 'Dueling_', False: ''}[dueling] + 'DQN')

    if minatar:
        model.add(Conv2D(16, 3, strides=1, activation='relu',
                  input_shape=input_shape, name='Input'))
        model.add(Flatten())
    elif embedding is not None:
        model.add(Embedding(embedding[0], embedding[1],
                  input_shape=input_shape, name='Input'))
        model.add(Reshape(embedding[2]))
        model.add(Dense(layers[0], activation="relu", name='layer_0'))
    else:
        model.add(Dense(layers[0], input_shape=input_shape,
                  activation="relu", name='Input'))
    for i, layer in enumerate(layers[1:]):
        model.add(Dense(layer, activation="relu", name='layer_' + str(i + 1)))
    if dueling:
        model.add(Dense(output_shape + 1,
                  activation='linear', name="duelling_avg"))
        model.add(Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(
            a[:, 1:], axis=1, keepdims=True), output_shape=(output_shape,), name="actions"))
    else:
        model.add(Dense(output_shape, activation='linear', name="actions"))
    return model


class PolicyDiscreet:
    """
    Epsilon greedy policy for a deiscrete action
    """

    def __init__(self, actions, epsilon, espilon_decay, epsilon_min):
        self.actions = actions
        self.epsilon = epsilon
        self.espilon_decay = espilon_decay
        self.epsilon_min = epsilon_min

    def __call__(self, model, state, training=False, warmup=False):
        """
        Return the action for the given state from a model.
        """

        if not warmup:
            self.epsilon = max(
                self.epsilon_min, self.epsilon * self.espilon_decay)
        if warmup or (training and (np.random.rand() <= self.epsilon)):
            return np.random.choice(self.actions)
        else:
            return np.argmax(model.predict(np.expand_dims(state, axis=0)))


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
        indexes = np.random.choice(
            np.arange(self.len), size=batch_size, replace=False)
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
        self.minatar = isinstance(env, Environment)

    def compile(self):
        """
        Compile the model.
        """
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
        self.target_model.compile(
            optimizer=Adam(learning_rate=1e-3), loss='mse')
        self.model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
        self.model.summary()

    def update_target_model_hard(self):
        """
        Update the target model weight to match the online model
        """
        self.target_model.set_weights(self.model.get_weights())

    def backward(self, batch_size, gamma, n_steps, weights_path):
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
            if weights_path is not None:
                self.save_weights(weights_path)

        return hist

    def train(self, max_steps=10000, batch_size=32, gamma=0.95, render=False, weights_path=None, checkpoint_path=None):
        """
        Train the model.
        """
        n_steps = 0
        n_episodes = 0
        episodes_rewards = []
        episodes_steps = []
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                checkpoint_confg = json.load(f)
                n_steps = checkpoint_confg['n_steps']
                n_episodes = checkpoint_confg['n_episodes']
                episodes_rewards = checkpoint_confg['episodes_rewards']
                episodes_steps = checkpoint_confg['episodes_steps']
                self.policy.epsilon = checkpoint_confg['policy_epsilon']
                for entry in checkpoint_confg['memory']:
                    self.memory.add((np.array(entry[0]), entry[1], entry[2], np.array(entry[3]), entry[4]))
        history = []
        previous_step_save = n_steps
        while(n_steps <= max_steps + self.warmup_steps):
            start_time = datetime.now()
            n_episodes += 1
            episode_reward = 0
            steps = 0
            if self.minatar:
                self.env.reset()
                state = np.array(self.env.state()).astype(np.int32)
            else:
                state = np.array(self.env.reset())
            done = False
            if render:
                self.env.render()
            while(not done):
                n_steps += 1
                steps += 1
                action = self.policy(self.model, state, training=True, warmup=(
                    n_steps <= self.warmup_steps))
                if self.minatar:
                    reward, done = self.env.act(action)
                    next_state = self.env.state()
                    next_state = np.array(next_state).astype(np.int32)
                else:
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.array(next_state)
                episode_reward += reward
                # Add to memory the experience
                self.memory.add((state, action, reward, next_state, done))

                if self.warmup_steps < n_steps:
                    history.append(self.backward(
                        batch_size, gamma, n_steps, weights_path))

                state = next_state
                if render:
                    self.env.render()
                if n_steps > max_steps + self.warmup_steps:
                    break
            episodes_rewards.append(episode_reward)
            episodes_steps.append(n_steps)

            if checkpoint_path is not None and n_steps - previous_step_save >= 1000:
                previous_step_save = n_steps
                with open(checkpoint_path, 'w') as f:
                    json.dump({
                        'n_steps': n_steps,
                        'n_episodes': n_episodes,
                        'policy_epsilon': self.policy.epsilon,
                        'episodes_rewards': episodes_rewards,
                        'episodes_steps': episodes_steps,
                        'memory': list(self.memory.buffer)
                    }, f, cls=NumpyEncoder)

            end_time = datetime.now()
            print("Total Steps: {}\t\tEpisode: {}\t\tReward: {},\t\tEpsilon: {},\t\tSteps: {},\t\tTime: {}".format(
                n_steps, n_episodes, episode_reward, self.policy.epsilon, steps, (end_time - start_time).total_seconds()))
        episodes_rewards.pop()
        episodes_steps.pop()
        return history, episodes_rewards, episodes_steps

    def test(self, n_tests=1, success_average=1, render=False):
        """
        Test the model
        """
        total_reward_array = np.zeros(n_tests)
        for test in range(n_tests):
            start_time = datetime.now()
            steps = 0
            episode_reward = 0
            if self.minatar:
                self.env.reset()
                state = np.array(self.env.state())
            else:
                state = np.array(self.env.reset())
            done = False
            if render:
                self.env.render()
            while(not done):
                steps += 1
                action = self.policy(
                    self.model, state, training=False, warmup=False)
                if self.minatar:
                    reward, done = self.env.act(action)
                    next_state = self.env.state()
                else:
                    next_state, reward, done, _ = self.env.step(action)
                next_state = np.array(next_state)
                episode_reward += reward
                state = next_state
                if render:
                    self.env.render()
            total_reward_array[test] = episode_reward
            end_time = datetime.now()
            print("Test {}\t\tReward: {},\t\tSteps: {},\t\tTime: {}".format(
                test + 1, episode_reward, steps, (end_time - start_time).total_seconds()))
        average = float(total_reward_array.sum())/float(n_tests)
        print("Average Success: {} - Experience {}".format(average,
              {True: "Success", False: "Failed"}[average >= success_average]))

    def save(self, path):
        """
        Save the online model
        """
        self.model.save(path)

    def save_weights(self, path):
        """
        Save the online model weights
        """
        self.model.save_weights(path)

    def load(self, path):
        """
        Load the model and compile both models
        """
        self.model = load_model(path)
        self.compile()

    def load_weights(self, path):
        """
        Load the weights of the online model
        """
        self.model.load_weights(path)
        self.compile()


def plot_rewards(rewards, name, path):
    """
    Plot the rewards
    """
    plt.plot(rewards)
    plt.title(name)
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.savefig(path)
    plt.close()


def get_updated_reward_from_config(rewards, config):
    if config['avg_plot']:
        plot_reward = []
        w = config['avg_window_plot']
        w_2 = w // 2
        for i in range(len(rewards)):
            s = max(0, i - w_2)
            e = min(len(rewards), i + w_2)
            plot_reward.append(sum(rewards[s:e]) / (e - s + 1))
        return plot_reward
    else:
        return rewards


def plot_avg_rewards(results, config):
    """
    Plot the rewards
    """

    # cluster = []
    # for i in range(len(rewards)):
    #     cluster.append(i % config['avg_window_plot'])

    # r = pd.DataFrame(list(zip(cluster, rewards)), columns=['Episode', 'Reward'])
    # sns.lineplot(x="Episode", y="Reward", data=r , ci='sd')
    plot_rewards = get_updated_reward_from_config(results["train_rewards"], config)
    plt.plot(results["train_steps"], plot_rewards)
    plt.title(config['name'])
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.savefig(config['plot_path'])
    plt.close()


def plot_compare_rewards(rewards_list, config_list, path):
    """
    Plot the comparaison rewards
    """
    labels = []
    for results, config in zip(rewards_list, config_list):
        plot_rewards = get_updated_reward_from_config(results["train_rewards"], config)
        labels.append(config['name'])
        plt.plot(results["train_steps"], plot_rewards)
    title = " ".join(labels)
    plt.legend(labels)
    plt.title("Compare " + title)
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.savefig(path)
    plt.close()


def run_from_config(config, render_tests=False):
    """
    Run the agent from a config file.
    """

    print("Running from config: {}".format(config['env']))
    np.random.seed(config['seed'])
    memory_config = config['memory_config']
    policy_config = config['policy_config']
    agent_config = config['agent_config']
    train_config = config['train_config']
    test_config = config['test_config']
    env = create_gym_env(config['env'], config['seed'])
    if "action_space" in config:
        n = config['action_space']
    else:
        n = env.action_space.n
    model = create_model(config['input_shape'], n, layers=config['layers'], dueling=config['dueling'],
                         embedding=config['embedding'], minatar=config['env'].startswith('minatar'))
    memory = Memory(max_size=memory_config['max_size'])
    policy = PolicyDiscreet(
        n, policy_config["epsilon"], policy_config['epsilon_decay'], policy_config['epsilon_min'])
    agent = Agent(model, env, memory, warmup_steps=agent_config['warmup_steps'],
                  target_model_update=agent_config['target_model_update'], policy=policy, ddqn=True)
    if config['model_path'] is not None and os.path.exists(config['model_path']):
        print("Model is already trained, loading the weights from {}".format(
            config['model_path']))
        agent.load(config['model_path'])
        policy.epsilon = policy.epsilon_min
        history = []
        with open(config['rewards_path'], "r") as f:
            trains_results = json.load(f)
    else:
        if config['weight_path'] is not None and os.path.exists(config['weight_path']):
            agent.load_weights(config['weight_path'])
        else:
            agent.compile()
        history, train_rewards, train_steps = agent.train(max_steps=train_config['max_steps'], batch_size=train_config['batch_size'],
                                             gamma=train_config['gamma'], weights_path=config['weight_path'], checkpoint_path=config['checkpoint_path'])
        trains_results = {"train_rewards":train_rewards,"train_steps":train_steps}
        agent.save(config['model_path'])
        with open(config['rewards_path'], "w") as f:
            json.dump({"train_rewards":train_rewards,"train_steps":train_steps}, f)
    agent.test(n_tests=test_config['n_tests'], render=render_tests)

    return history, trains_results, agent


def render_from_config(config):
    """
    Run the agent from a config file.
    """
    print("Running from config: {}".format(config['env']))
    memory_config = config['memory_config']
    policy_config = config['policy_config']
    agent_config = config['agent_config']
    env = create_gym_env(config['env'], config['seed'])
    if "action_space" in config:
        n = config['action_space']
    else:
        n = env.action_space.n
    model = create_model(config['input_shape'], n, layers=config['layers'], dueling=config['dueling'],
                         embedding=config['embedding'], minatar=config['env'].startswith('minatar'))
    memory = Memory(max_size=memory_config['max_size'])
    policy = PolicyDiscreet(
        n, policy_config["epsilon"], policy_config['epsilon_decay'], policy_config['epsilon_min'])
    policy.epsilon = policy.epsilon_min
    agent = Agent(model, env, memory, warmup_steps=agent_config['warmup_steps'],
                  target_model_update=agent_config['target_model_update'], policy=policy, ddqn=True)
    if config['model_path'] is not None and os.path.exists(config['model_path']):
        print("Model is already trained, loading the weights from {}".format(
            config['model_path']))
        agent.load(config['model_path'])
    else:
        raise Exception("Model not found")
    agent.test(n_tests=1, render=True)
    return
