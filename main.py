# Credits to Machine Learning with Phil youtube channel

import gym
from agent import Agent
import numpy as np


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = Agent(gamma=0.95, epsilon=1, batch_size=128, n_actions=env.action_space.n, eps_end=0.01,
                  input_dims=4, lr=0.001)
    scores, eps_history = [], []
    n_games = 500

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            new_observation, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, new_observation, done)
            agent.learn()
            observation = new_observation
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        print(f'episode {i}, score {score}, average score: {avg_score}, epsilon= {agent.epsilon}')