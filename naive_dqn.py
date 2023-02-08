"""
Naive Deep Q Network

Source: https://www.youtube.com/watch?v=5fHngyN8Qhw
"""

from keras.layers import Dense, Input
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.utils import to_categorical
import tensorflow as tf
import numpy as np


def build_dqn(learning_rate, n_actions, input_shape, layer_dims):
    """
    Builds a fully connected dense network for the purpose of evaluating the
    Agent's actions

    :param learning_rate: network learning rate
    :param n_actions: the Agent's action space
    :param input_shape: the shape of the input information from the environment,
        the state shape
    :param layer_dims: the sizes of the layers in the evaluator network
    :return:
    """
    model = Sequential()

    model.add(Input(shape=input_shape))
    for layer_size in layer_dims:
        model.add(Dense(layer_size, activation="relu"))
    model.add(Dense(n_actions))

    # Adam optimizer with mean squared error loss function
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    return model


class ReplayBuffer(object):

    def __init__(self, memory_size, input_shape, n_actions, discrete=False):
        """
        Replay Buffer stores information about the network's actions and their effects
        on the network's environment.

        :param memory_size: maximum memory of the network
        :param input_shape: shape of the information about the current environment state
        :param n_actions: action space cardinality
        :param discrete: defines the type of action space - if discrete then we
            can use a one-hot encoding, if not then we must store the agent's vector output
        """
        # Least recently updated memory index
        self.memory_index = 0

        # Network parameters
        self.memory_size = memory_size
        self.n_actions = n_actions
        self.state_memory_shape = [self.memory_size, input_shape]
        self.action_memory_shape = [self.memory_size, self.n_actions]
        self.discrete = discrete

        # Stores environment information
        self.state_memory = np.zeros(self.state_memory_shape)
        self.new_state_memory = np.zeros(self.state_memory_shape)

        self.action_memory = np.zeros(self.action_memory_shape)

        self.reward_memory = np.zeros(self.memory_size)

        # At the terminal state when the episode is over, ignore future rewards
        self.terminal_memory = np.zeros(self.memory_size)

    def store_transition(self, state, action, reward, _state, done):
        """
        Stores the information for a particular action taken by the network.

        :param state: the state of the environment before the network acted
        :param action: the action the network performed on the environment
        :param reward: the network evaluation of the value of the action
        :param _state: the state of the environment after it was acted on
        :param done: True if there can be no more actions performed on the
            environment.
        """
        # Least recently updated memory index
        index = self.memory_index % self.memory_size

        # Store the information about the network's effect on its environment
        self.state_memory[index] = state
        self.new_state_memory[index] = _state

        self.reward_memory[index] = reward

        # The environment will return done=True when the Agent can no longer act.
        # For our reward function we want to add possible future rewards as long
        # as the Agent is not done. So we multiply by the terminal memory and we
        # need to flip the done bits
        self.terminal_memory[index] = 1 - done

        if self.discrete:
            # one hot encode the action
            self.action_memory[index] = to_categorical(action, num_classes=self.n_actions)
        else:
            # store the autoencoded vector containing state information
            self.action_memory[index] = action

        self.memory_index += 1

    def sample_buffer(self, batch_size):
        """
        Randomly chooses batch_size elements from state, new state, action,
        reward, and terminal memory

        :param batch_size: number of elements of memory to be returned
        :return: A random sampling of the buffer memory
        """
        # The highest index that holds state information
        highest_memory_index = min(self.memory_index, self.memory_size)
        batch = np.random.choice(highest_memory_index, batch_size)

        return self.state_memory[batch], \
               self.new_state_memory[batch], \
               self.reward_memory[batch], \
               self.action_memory[batch], \
               self.terminal_memory[batch]


class Agent(object):

    def __init__(self, learning_rate, gamma, n_actions, epsilon, batch_size,
                 input_dims, discrete=False, q_network_layers=[256, 256],
                 epsilon_decrement=0.996, epsilon_min=0.01,
                 memory_size=1000000, save_file='dqn_model.h5'):
        """
        Class modeling an Agent performing actions on a given environment using
        an epsilon-greedy approach where epsilon is the exploration parameter

        :param learning_rate: network learning rate
        :param gamma: scaling factor for future rewards
            ( > 1 : long term gratification, < 1 : short term gratification)
        :param n_actions: action space for the network
        :param discrete: defines the type of action space - if discrete then we
            can use a one-hot encoding, if not then we must store the action as a vector

            Ex:
                Movement in a discrete space Up, Down, Left, Right can be encoded
                [1, 0, 0, 0] is one space up, [0, 0, 1, 0] is one space left.

                Movement in a continuous space Up, Down, Left, Right must be left as
                a vector: [0.5, 0.25, -.6, .3] would translate to .25 units up and .9
                units to the right

        :param q_network_layers: the shape of the evaluation network
        :param epsilon: probability the agent performs a random action
            (exploration parameter)
        :param batch_size: training batch size
        :param input_dims: dimension of state information about the environment
        :param epsilon_decrement: rate at which epsilon decreases during training
        :param epsilon_min: minimum rate of random actions
        :param memory_size: defines replay buffer size
        :param save_file: filename to save the trained network to
        """
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decrement = epsilon_decrement
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.model_file = save_file

        self.n_actions = n_actions
        self.discrete = discrete

        # replay memory
        self.memory = ReplayBuffer(memory_size, input_dims, self.n_actions, self.discrete)

        # evaluation network
        self.q_eval = build_dqn(learning_rate, n_actions, input_dims, q_network_layers)

    def remember(self, state, action, reward, _state, done):
        self.memory.store_transition(state, action, reward, _state, done)

    def choose_action(self, state):
        state = tf.expand_dims(state, axis=0)

        # ----------------------------------------------------------------------------
        # BEGIN EPSILON GREEDY
        # ----------------------------------------------------------------------------
        if np.random.random() < self.epsilon:
            # Exploration action
            if self.discrete:
                # one hot encode actions and choose one
                action = np.random.randint(self.n_actions)
            else:
                # sample a uniform distribution 0.0 to 1.0
                action = np.random.uniform(size=self.n_actions)
        else:
            action = self.q_eval.predict(state, verbose=0)
            if self.discrete:
                action = np.argmax(action)

        return action

    def learn(self):
        # Wait until the Agent has played around with its environment before beginning
        # learning (until at least batch_size actions have been performed)
        if self.memory.memory_index < self.batch_size:
            return

        # get a batch of Agent actions to train on
        state, _state, reward, action, done = self.memory.sample_buffer(self.batch_size)

        # Get the evaluation network's predictions before and after the Agent acts
        #
        # For discrete action spaces, each prediction is a vector of predicted
        # reward values given the Agent chooses the action at that index.
        # prediction[action] = predicted reward
        #
        # For continuous action spaces, each prediction is the next action to take
        # in order to maximize rewards
        reward_prediction = self.q_eval.predict(state, verbose=0)
        next_reward = self.q_eval.predict(_state, verbose=0)

        # Calculate the target reward
        #
        # For discrete action spaces:
        # target_reward[action] = reward + gamma * max( next_reward )
        #
        # For continuous action spaces:
        # TODO: find a good reward function. placeholder for now
        target_reward = reward_prediction.copy()

        if self.discrete:
            # find the action indices so that we can update target rewards based
            # on the action the Agent took
            #
            # We can do this by taking a dot product of the action space with the
            # one hot vectors.
            # Ex. action space of size 6, action took was 3
            #     action space vector u = [0, 1, 2, 3, 4, 5]
            #     action one hot v = [0, 0, 0, 1, 0, 0]
            #     u * v = 3

            # action_values = tf.constant(tf.range(self.n_actions), shape=self.n_actions)
            # action_indices = tf.tensordot(action, action_values)

            # We can also do this with tf.math.argmax across axis 1
            action_indices = tf.math.argmax(action, axis=-1)

            # Need batch_index to index rewards
            batch_index = tf.constant(tf.range(self.batch_size), shape=self.batch_size)

            # only update target_reward if the Agent can continue acting on the environment
            # (if the Agent is not done)
            target_reward[batch_index, action_indices] = reward + self.gamma * np.max(next_reward, axis=-1) * done
        else:
            # Placeholder for continuous action space reward function
            pass

        # train the evaluation network on the target reward
        _ = self.q_eval.fit(state, target_reward, verbose=0)

        self.epsilon *= self.epsilon_decrement
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)
