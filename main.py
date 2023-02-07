from naive_dqn import Agent
import numpy as np
from utils import plotLearning
import gym

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    n_games = 500
    agent = Agent(learning_rate=0.0005, gamma=0.99, n_actions=4, epsilon=1.0,
                  batch_size=64, input_dims=8, discrete=True)

    # agent.load_model()
    scores = []
    epsilon_history = []

    for _ in range(n_games):
        done = False
        score = 0

        # reset the environment at the start of each game
        state, info = env.reset()
        while not done:
            action = agent.choose_action(state)

            # Take an observation of the environment state
            _state, reward, done, truncated, info = env.step(action)

            # update our score based on the reward
            score += reward

            # store the observation in the agent memory
            agent.remember(state, action, reward, _state, done)

            # learn from the agent memory
            agent.learn()

            # set up for next step
            state = _state

        scores.append(score)
        epsilon_history.append(agent.epsilon)

        avg_score = np.mean(scores)

        print('Game number {}:\n\tscore {}\n\taverage score {}\n'.format(_ + 1, score, avg_score))

        if _ % 10 == 0 and _ > 0:
            agent.save_model()

    plt_fname = 'lunarlander.png'
    x = [i + 1 for i in range(n_games)]
    plotLearning(x, scores, epsilon_history, plt_fname)
