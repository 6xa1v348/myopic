import enum
import math

import gym_jumping_task
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

import gym_helpers


class OBSTACLE_COLORS(enum.Enum):   # pylint: disable=invalid-name
    RED = 0
    GREEN = 1

JUMP_DISTANCE = 13


def stack_obs(observation_list):
    zero_observation = [np.zeros_like(observation_list[0])]
    return np.stack(
        (zero_observation + observation_list[:-1], observation_list),
        axis=1)


def generate_optimal_trajectory(obstacle_position, floor_height):
    env = gym_helpers.create_gym_environment('jumping-task')
    env.reset()
    initial_obs = env.unwrapped._reset( # pylint: disable=protected-access
        obstacle_position=obstacle_position,
        floor_height=floor_height)
    terminal = False
    obs = initial_obs
    observations, rewards, actions = [], [], []
    counter = 0
    while not terminal:
        counter += 1
        if (counter == obstacle_position - JUMP_DISTANCE):
            action = 1
        else:
            action = 0
        next_obs, reward, terminal, _ = env.step(action)
        observations.append(obs)
        rewards.append(reward)
        actions.append(action)
        obs = next_obs
    observations.append(obs)
    assert sum(rewards) >= counter + 1, 'Trajectory not optimal!'
    return (observations, actions, rewards)


def generate_imitation_data(min_obstacle_position=20,
                             max_obstacle_position=45,
                             min_floor_height=10,
                             max_floor_height=20):
    data = {}
    for position in range(min_obstacle_position, max_obstacle_position + 1):
        data[position] = {}
        for height in range(min_floor_height, max_floor_height + 1):
            observations, actions, rewards = generate_optimal_trajectory(
                position, height)
            observations = stack_obs(observations)
            data[position][height] = (
                observations[:-1], actions, rewards, observations[1:])
    return data


def generate_training_positions(min_obstacle_position=20,
                                max_obstacle_position=45,
                                min_floor_height=10,
                                max_floor_height=20,
                                positions_train_diff=5,
                                heights_train_diff=5,
                                random_tasks=False,
                                seed=0):
    if random_tasks:
        obstacle_positions = list(
            range(min_obstacle_position, max_obstacle_position + 1))
        floor_heights = list(
            range(min_floor_height, max_floor_height + 1))
        num_positions = (len(obstacle_positions) // positions_train_diff) + 1
        num_heights = (len(floor_heights) // heights_train_diff) + 1
        num_train_positions = num_positions * num_heights
        np.random.seed(seed)
        obstacle_positions_train = np.random.choice(
            obstacle_positions, size=num_train_positions)
        floor_heights_train = np.random.choice(
            floor_heights, size=num_train_positions)
        training_positions = list(
            zip(obstacle_positions_train, floor_heights_train))
    else:
        obstacle_positions_train = list(
            range(min_obstacle_position, max_obstacle_position + 1, positions_train_diff))
        floor_heights_train = list(
            range(min_floor_height, max_floor_height + 1, heights_train_diff))
        training_positions = []
        for pos in obstacle_positions_train:
            for height in floor_heights_train:
                training_positions.append((pos, height))
    return training_positions


def neighbor_indices(x, y, max_x, max_y):
    valid_indices = []
    for index in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
        is_x_valid = (0 <= index[0]) and (index[0] < max_x)
        is_y_valid = (0 <= index[1]) and (index[1] < max_y)
        if is_x_valid and is_y_valid:
            valid_indices.append(index)
    return valid_indices


def generate_validation_positions(training_positions,
                                  min_obs_position,
                                  min_floor_height,
                                  num_positions,
                                  num_heights):
    validation_positions = []
    for (pos, height) in training_positions:
        pos_index = pos - min_obs_position
        height_index = height - min_floor_height
        valid_indices = neighbor_indices(
            pos_index, height_index, num_positions, num_heights)
        for valid_pos_index, valid_height_index in valid_indices:
            validation_positions.append((valid_pos_index + min_obs_position,
                                         valid_height_index + min_floor_height))
    return list(set(validation_positions))


def training_data(imitation_data, training_positions):
    obs, acs, next_obs = [], [], []
    for (position, height) in training_positions:
        o, a, _, p = imitation_data[position][height]
        obs.extend(o)
        acs.extend(a)
        next_obs.extend(p)
    obs = np.array(obs, dtype=np.float32)
    acs = np.array(acs, dtype=np.int64)
    next_obs = np.array(next_obs, dtype=np.float32)
    return obs, acs, next_obs


def sample_train_pair_index(training_positions):
    index1, index2 = 0, 0
    while index1 == index2:
        index1, index2 = np.random.randint(0, high=len(training_positions), size=2)
    pos1, pos2 = training_positions[index1], training_positions[index2]
    return pos1, pos2


def generate_optimal_data_tuple(imitation_data, training_positions):
    (pos1, height1), (pos2, height2) = sample_train_pair_index(training_positions)
    return (imitation_data[pos1][height1],
            imitation_data[pos2][height2])


def create_balanced_dataset(obs, acs, next_obs, batch_size):
    acs = acs.astype(np.int64)
    class_sample_count = [len(np.where(acs == ac)[0]) for ac in np.unique(acs)]
    weight = 1. / np.array(class_sample_count)
    samples_weight = torch.from_numpy(
        np.array([weight[ac] for ac in acs])).double()
    num_samples = math.ceil(len(samples_weight) / batch_size) * batch_size
    sampler = WeightedRandomSampler(samples_weight, num_samples)
    dataset = TensorDataset(
        torch.tensor(obs).float(),
        torch.tensor(acs).long(),
        torch.tensor(next_obs).float())
    return DataLoader(dataset, batch_size=batch_size, num_workers=1, sampler=sampler)
