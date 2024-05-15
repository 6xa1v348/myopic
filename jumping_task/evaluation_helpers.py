import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style('white')


def create_evaluation_grid(model, imitation_data, mc_samples):
    model.train(False)
    obstacle_positions = sorted(imitation_data.keys())
    floor_heights = sorted(imitation_data[obstacle_positions[0]].keys())
    evaluation_grid = np.zeros((len(obstacle_positions), len(floor_heights)))
    for i, pos in enumerate(obstacle_positions):
        for j, height in enumerate(floor_heights):
            observations, optimal_actions, _, _ = imitation_data[pos][height]
            observations = torch.tensor(observations).float().to(model.device)
            with torch.no_grad():
                predictions = torch.softmax(model.actor(observations), dim=-1)
                for _ in range(mc_samples - 1):
                    predictions += torch.softmax(model.actor(observations), dim=-1)
                predictions /= mc_samples
            greedy_actions = torch.argmax(predictions, dim=-1).cpu().numpy()
            action_diff = greedy_actions - np.array(optimal_actions)
            argmax_val = np.argmax(optimal_actions)
            binary_mask = np.arange(len(optimal_actions)) <= argmax_val
            is_optimal = sum(binary_mask * np.abs(action_diff)) == 0
            evaluation_grid[i][j] = is_optimal
    return evaluation_grid


def compute_norm(model, imitation_data):
    model.train(False)
    obstacle_positions = sorted(imitation_data.keys())
    floor_heights = sorted(imitation_data[obstacle_positions[0]].keys())
    norms = []
    for pos in obstacle_positions:
        for height in floor_heights:
            observations, _, _, _ = imitation_data[pos][height]
            observations = torch.tensor(observations).float().to(model.device)
            with torch.no_grad():
                norms.append(model.compute_norm(observations))
    return np.nanmean(norms)


def num_solved_tasks(evaluation_grid,
                     training_positions,
                     validation_positions,
                     min_obs_position,
                     min_floor_height):
    solved_envs = {'train': 0, 'test': 0}
    if validation_positions:
        solved_envs['validation'] = 0

    num_positions, num_heights = evaluation_grid.shape
    is_train_or_validation = np.zeros_like(evaluation_grid, dtype=np.int32)

    for (pos, height) in training_positions:
        pos_index = pos - min_obs_position
        height_index = height - min_floor_height
        is_train_or_validation[pos_index][height_index] = 1

    for (pos, height) in validation_positions:
        pos_index = pos - min_obs_position
        height_index = height - min_floor_height
        is_train_or_validation[pos_index][height_index] = 2

    for pos_index in range(num_positions):
        for height_index in range(num_heights):
            if is_train_or_validation[pos_index][height_index] == 1:
                solved_envs['train'] += evaluation_grid[pos_index][height_index]
            elif is_train_or_validation[pos_index][height_index] == 2:
                solved_envs['validation'] += evaluation_grid[pos_index][height_index]
            else:
                solved_envs['test'] += evaluation_grid[pos_index][height_index]
    return solved_envs


def plot_evaluation_grid(evaluation_grid,
                         training_positions,
                         min_obs_position,
                         min_floor_height):
    fig, ax = plt.subplots(figsize=(6, 4))
    grid_x, grid_y = evaluation_grid.shape
    extent = (0, grid_x, 0, grid_y)
    ax.imshow(evaluation_grid.T, extent=extent, origin='lower', vmin=0, vmax=1)

    x_ticks = np.arange(grid_x)
    y_ticks = np.arange(grid_y)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.tick_params(labelbottom=False, labelleft=False)

    # Loop over data dimensions and create text annotations
    for (pos, height) in training_positions:
        pos_index = pos - min_obs_position
        height_index = height - min_floor_height
        ax.text(
            pos_index + 0.5,
            height_index + 0.5,
            'T',
            ha='center',
            va='center',
            color='r')
    ax.grid(color='w', linewidth=1)
    fig.tight_layout()
    return fig
