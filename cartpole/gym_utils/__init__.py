import gym

from .continuous_cartpole import ContinuousCartPoleEnv

gym.register(id='ContinuousCartpole-v0',
             entry_point='gym_utils.continuous_cartpole:ContinuousCartPoleEnv',
             max_episode_steps=200)