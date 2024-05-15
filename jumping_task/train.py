import os.path as osp
import time
import argparse
from collections import defaultdict

import numpy as np
import torch

from utils import set_seed_everywhere, mkdir
from logger import Logger
from agents import *
import data_helpers
import evaluation_helpers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--work-dir', type=str, default='.')

    parser.add_argument('--training-epochs', type=int, default=2000)
    parser.add_argument('--log-every-n-epochs', type=int, default=10)
    parser.add_argument('--evaluate-every-n-epochs',type=int, default=20)

    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--grid', type=str, default='wide', choices=['wide', 'narrow'])
    parser.add_argument('--random-tasks', action='store_true')
    parser.add_argument('--no-validation', action='store_true')

    parser.add_argument('--agent', type=str, choices=['baseline', 'pse', 'myopic'])
    parser.add_argument('--multi-positive', action='store_true')
    parser.add_argument('--projection', action='store_true')
    parser.add_argument('--rand-conv', action='store_true')
    parser.add_argument('--kernel-size', type=int, default=2)
    parser.add_argument('--max-norm', action='store_true')
    parser.add_argument('--encoder-feature-dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=5.0)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--soft-coupling-temperature', type=float, default=0.01)
    parser.add_argument('--max-grad-norm', type=float, default=0.)

    args = parser.parse_args()

    args.min_obstacle_grid = 20
    args.max_obstacle_grid = 45
    args.min_floor_grid = 10
    args.max_floor_grid = 20

    if args.random_tasks or args.grid == 'wide':
        args.min_obstacle_position = 20
        args.max_obstacle_position = 45
        args.min_floor_height = 10
        args.max_floor_height = 20
        args.positions_train_diff = 5
        args.heights_train_diff = 5
    else:
        args.min_obstacle_position = 28
        args.max_obstacle_position = 38
        args.min_floor_height = 13
        args.max_floor_height = 17
        args.positions_train_diff = 2
        args.heights_train_diff = 2

    if args.agent == 'baseline':
        assert args.alpha == 0.
    elif args.agent == 'myopic':
        assert args.gamma == 0.
    
    return args


def train(args):
    set_seed_everywhere(args.seed)
    if args.device is None:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%s' % args.device)
    print('Using %s device' % device)

    mkdir(args.work_dir)
    if args.no_validation:
        model_dir = mkdir(osp.join(args.work_dir, 'model'))
        grid_dir = mkdir(osp.join(args.work_dir, 'grid'))
    
    # make dataset
    imitation_data = data_helpers.generate_imitation_data(
        min_obstacle_position=args.min_obstacle_grid,
        max_obstacle_position=args.max_obstacle_grid,
        min_floor_height=args.min_floor_grid,
        max_floor_height=args.max_floor_grid)
    training_positions = data_helpers.generate_training_positions(
        min_obstacle_position=args.min_obstacle_position,
        max_obstacle_position=args.max_obstacle_position,
        min_floor_height=args.min_floor_height,
        max_floor_height=args.max_floor_height,
        positions_train_diff=args.positions_train_diff,
        heights_train_diff=args.heights_train_diff,
        random_tasks=args.random_tasks)
    
    num_positions = args.max_obstacle_grid - args.min_obstacle_grid + 1
    num_heights = args.max_floor_grid - args.min_floor_grid + 1

    if args.no_validation:
        validation_positions = []
    else:
        validation_positions = data_helpers.generate_validation_positions(
            training_positions,
            min_obs_position=args.min_obstacle_grid,
            min_floor_height=args.min_floor_grid,
            num_positions=num_positions,
            num_heights=num_heights)
    num_train = len(training_positions)
    num_validation = len(validation_positions)
    num_test = (num_positions * num_heights) - num_train - num_validation

    obs_train, acs_train, next_obs_train = data_helpers.training_data(imitation_data, training_positions)
    trainloader = data_helpers.create_balanced_dataset(
        obs_train, acs_train, next_obs_train, batch_size=args.batch_size)
    
    # make agent
    if args.agent in ['baseline', 'pse']:
        Network = PSE
    elif args.agent == 'myopic':
        if args.multi_positive:
            Network = MultiPSE
        else:
            Network = PSE
    else:
        raise ValueError('Unknown agent %s' % args.agent)
    
    agent = Network(obs_shape=obs_train[0].shape,
                    num_actions=2,
                    device=device,
                    encoder_feature_dim=args.encoder_feature_dim,
                    kernel_size=args.kernel_size,
                    projection=args.projection,
                    rand_conv=args.rand_conv,
                    lr=args.lr,
                    alpha=args.alpha,
                    gamma=args.gamma,
                    temperature=args.temperature,
                    soft_coupling_temperature=args.soft_coupling_temperature,
                    max_grad_norm=args.max_grad_norm,
                    use_coupling_weights=True,
                    max_norm=args.max_norm)
    
    # preparation
    L = Logger(log_dir=args.work_dir)
    start_epoch = 1
    start_time = time.time()
    mc_samples = 5 if args.rand_conv else 1

    for epoch in range(start_epoch, args.training_epochs + 1):
        # update learning rate
        agent.decay_learning_rate(epoch)

        epoch_losses = defaultdict(list)
        for obs, acs, _ in trainloader:
            obs = obs.to(device)
            acs = acs.to(device)
            if args.alpha > 0:
                optimal_data_tuple = data_helpers.generate_optimal_data_tuple(
                    imitation_data, training_positions)
            else:
                optimal_data_tuple = None
            
            losses = agent.update(obs, acs, optimal_data_tuple=optimal_data_tuple)
            for key, value in losses.items():
                epoch_losses[key].append(value)
    
        # log periodically
        if epoch % args.log_every_n_epochs == 0:
            L.log('train/epoch', epoch)
            if args.seed: L.log('train/seed', args.seed)
            for key, value in epoch_losses.items():
                L.log('train/%s' % key, np.nanmean(value))

        # evaluate periodically
        if epoch % args.evaluate_every_n_epochs == 0:
            evaluation_grid = evaluation_helpers.create_evaluation_grid(
                agent, imitation_data, mc_samples=mc_samples)
            solved_envs = evaluation_helpers.num_solved_tasks(
                evaluation_grid, training_positions, validation_positions,
                min_obs_position=args.min_obstacle_grid, min_floor_height=args.min_floor_grid)
            for key in solved_envs.keys():
                if key == 'train':
                    solved_envs[key] /= num_train
                elif key == 'validation':
                    solved_envs[key] /= num_validation
                elif key == 'test':
                    solved_envs[key] /= num_test
            L.log('eval/epoch', epoch)
            if not args.no_validation:
                L.log('eval/lr', args.lr)
                L.log('eval/alpha', args.alpha)
                if args.agent in ['myopic']: L.log('eval/temperature', args.temperature)
            if args.seed: L.log('eval/seed', args.seed)
            for key, value in solved_envs.items():
                L.log('eval/%s' % key, value)
            L.log('eval/duration', time.time() - start_time)
        L.dump()

    if args.no_validation:
        np.save(osp.join(grid_dir, str(args.seed)), evaluation_grid)
        agent.save(osp.join(model_dir, str(args.seed)))
    
    print('Completed in %.02f mins' % ((time.time() - start_time) / 60))
    if not args.no_validation: return solved_envs['validation']


if __name__ == "__main__":
    args = parse_args()
    train(args)
