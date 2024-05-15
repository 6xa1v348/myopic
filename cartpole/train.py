import os.path as osp
import time
import argparse

import gym
import gym_utils
from gym_utils import wrappers

import numpy as np
import torch

import utils
from agent import *
from logger import Logger


def parse_args():
    allowed_tasks = ['ContinuousCartpole-v0']
    allowed_agents = ['baseline', 'pse', 'bisim']

    parser = argparse.ArgumentParser()
    parser.add_argument('--domain-name', choices=allowed_tasks, help='Choose from ' + str(allowed_tasks))
    parser.add_argument('--sparsity-factor', default=1., type=float)
    parser.add_argument('--random-reward', action='store_true')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--work-dir', default='.', type=str)
    parser.add_argument('--device', type=int)
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--save-buffer', action='store_true')
    parser.add_argument('--multi-positive', action='store_true')
    
    parser.add_argument('--noisy-observation', action='store_true')
    parser.add_argument('--noisy-dims', default=0, type=int)
    parser.add_argument('--noise-std', default=1., type=float)
    parser.add_argument('--replay-buffer-capacity', default=50000, type=int)

    parser.add_argument('--agent', type=str, choices=allowed_agents)
    parser.add_argument('--init-steps', default=1000, type=int)
    parser.add_argument('--num-train-steps', default=60000, type=int)
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument('--hidden-dim', default=128, type=int)
    parser.add_argument('--k', default=3, type=int, help='Number of steps for inverse model.')
    
    parser.add_argument('--eval-freq', default=1000, type=int)
    parser.add_argument('--num-eval-episodes', default=10, type=int)

    parser.add_argument('--critic-lr', default=1e-3, type=float)
    parser.add_argument('--critic-tau', default=0.005, type=float)
    parser.add_argument('--critic-target-update-freq', default=2, type=int)
    parser.add_argument('--actor-lr', default=1e-3, type=float)
    parser.add_argument('--actor-log-std-min', default=-10., type=float)
    parser.add_argument('--actor-log-std-max', default=2., type=float)
    parser.add_argument('--actor-update-freq', default=2, type=int)
    parser.add_argument('--encoder-feature-dim', default=50, type=int)
    parser.add_argument('--encoder-lr', default=1e-3, type=float)
    parser.add_argument('--encoder-tau', default=0.005, type=float)
    parser.add_argument('--num-layers', default=2, type=int)
    parser.add_argument('--encoder-max-norm', action='store_true')
    parser.add_argument('--decoder-lr', default=1e-3, type=float)
    parser.add_argument('--decoder-update-freq', default=1, type=int)
    parser.add_argument('--decoder-weight-lambda', default=0., type=float)

    parser.add_argument('--c_R', default=1., type=float)
    parser.add_argument('--c_T', default=None, type=float)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init-temperature', default=0.01, type=float)
    parser.add_argument('--alpha-lr', default=1e-3, type=float)

    parser.add_argument('--intrinsic-reward-type', default='none', type=str,
                        choices=['none', 'forward_mean'])
    parser.add_argument('--intrinsic-reward-weight', type=float,
                        help='Weight on intrinsic reward (scale/eta).')
    parser.add_argument('--intrinsic-reward-max', type=float,
                        help='Maximum allowed intrinsic reward value (before weighting).')
    parser.add_argument('--latent-prior', default='none', type=str,
                        choices=['none', 'inverse_dynamics', 'tikhonov', 'unit_var', 'TK+ID'],
                        help='Additional regularization type on encoding.')
    parser.add_argument('--latent-prior-weight', type=float,
                        help='Loss weight on latent prior regularization.')
    
    args = parser.parse_args()
    if args.c_T is None:
        assert args.c_R == 1.
        args.c_T = args.discount
    assert (args.c_R <= 1. and args.c_T <= 1.)

    ### Task-specific defaults
    _defaults_w = {
        'SparsePendulum-v0':
            {'IRW': 0.10, 'IRM': 0.10, 'LPW_ID': 0.1, 'LPW_TK': 0.0005, 'LPW_UV': 0.001},
        'ContinuousCartpole-v0':
            {'IRW': 2.00, 'IRM': 0.10, 'LPW_ID': 1.0, 'LPW_TK': 0.0005, 'LPW_UV': 0.001},
        'MountainCarContinuous-v0':
            {'IRW': 20.0, 'IRM': 0.10, 'LPW_ID': 20.0, 'LPW_TK': 0.005, 'LPW_UV': 0.001},
    }
    if args.intrinsic_reward_weight is None and args.intrinsic_reward_type != 'none':
        args.intrinsic_reward_weight = _defaults_w[args.domain_name]['IRW']
    if args.intrinsic_reward_max is None and args.intrinsic_reward_type != 'none':
        args.intrinsic_reward_max = _defaults_w[args.domain_name]['IRM']

    if args.latent_prior_weight is None and args.latent_prior != 'none':
        _key = {'inverse_dynamics': 'LPW_ID', 'tikhonov': 'LPW_TK', 'unit_var': 'LPW_UV'}
        if args.latent_prior == 'TK+ID':
            _DW = _defaults_w[args.domain_name]
            args.latent_prior_weight = {'LPW_ID': _DW['LPW_ID'], 'LPW_TK': _DW['LPW_TK']}
        else:
            args.latent_prior_weight = _defaults_w[args.domain_name][_key[args.latent_prior]]
    
    print(args)
    return args


def evaluate(env, agent, num_episodes, L):
    episode_rewards = []
    norms = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            with utils.eval_mode(agent):
                norm = agent.compute_norm(obs)
                action = agent.select_action(obs, torch_mode=False)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            norms.append(norm)
        episode_rewards.append(episode_reward)
    L.log('eval/episode_reward', np.mean(episode_rewards))
    L.log('eval/norm', np.mean(norms))
    L.dump('eval')


def make_agent(obs_shape, action_shape, action_max, device, args):
    if args.agent == 'baseline':
        Agent = BaselineAgent
    elif args.agent == 'pse':
        if args.multi_positive:
            print('Using multiple positive pairs.')
            Agent = MultiPSEAgent
        else:
            Agent = PSEAgent
    elif args.agent == 'bisim':
        Agent = BisimAgent
    else:
        raise ValueError('Unknown agent: %s' % args.agent)
    agent = Agent(obs_shape=obs_shape,
                  action_shape=action_shape,
                  device=device,
                  hidden_dim=args.hidden_dim,
                  discount=args.discount,
                  init_temperature=args.init_temperature,
                  alpha_lr=args.alpha_lr,
                  actor_lr=args.actor_lr,
                  actor_action_max=action_max,
                  actor_log_std_min=args.actor_log_std_min,
                  actor_log_std_max=args.actor_log_std_max,
                  actor_update_freq=args.actor_update_freq,
                  critic_lr=args.critic_lr,
                  critic_tau=args.critic_tau,
                  critic_target_update_freq=args.critic_target_update_freq,
                  encoder_feature_dim=args.encoder_feature_dim,
                  encoder_lr=args.encoder_lr,
                  encoder_tau=args.encoder_tau,
                  num_layers=args.num_layers,
                  encoder_max_norm=args.encoder_max_norm)
    return agent


def main():
    args = parse_args()
    utils.set_seed_everywhere(args.seed)

    if args.domain_name in gym.envs.registry.env_specs.keys():
        if args.sparsity_factor < 1.:
            print('Building train env with sparsity factor %.02f.' % args.sparsity_factor)
            env = gym.make(args.domain_name, sparsity_factor=args.sparsity_factor)
        else:
            print('Building train env.')
            env = gym.make(args.domain_name)
        print('Building eval env.')
        eval_env = gym.make(args.domain_name)
        if args.random_reward:
            print('Wrapping train env with randomized reward wrapper.')
            env = wrappers.RandomReward(env)
        if args.noisy_observation and args.noisy_dims > 0:
            print('Wrapping train and eval env with noise distractor wrapper.')
            env = wrappers.NoisyObservationWrapper(env, noisy_dims=args.noisy_dims, noise_std=args.noise_std)
            eval_env = wrappers.NoisyObservationWrapper(eval_env, noisy_dims=args.noisy_dims, noise_std=args.noise_std)
    else:
        raise NotImplementedError
    
    utils.mkdir(args.work_dir)
    if args.save_model:
        model_dir = utils.mkdir(osp.join(args.work_dir, 'model'))
    if args.save_buffer:
        buffer_dir = utils.mkdir(osp.join(args.work_dir, 'buffer'))

    if args.device is None:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%s' % args.device)

    assert abs(env.action_space.low.min()) == env.action_space.high.max()
    action_max = env.action_space.high.max()

    use_IR = not (args.intrinsic_reward_type == 'none')
    IR_type = args.intrinsic_reward_type if use_IR else None
    w_IR = args.intrinsic_reward_weight
    IR_max = args.intrinsic_reward_max
    latprior = args.latent_prior
    w_latprior = args.latent_prior_weight
    if use_IR:
        assert w_IR > 0.0, 'Intrinsic reward weight must be positive.'
        print('Using intrinsic reward (%s): weight = %.02f' % (IR_type, w_IR))
    if not latprior == 'none':
        if type(w_latprior) is dict:
            for k in w_latprior.keys(): assert w_latprior[k] > 0.0
        else:
            assert w_latprior > 0.0, 'Latent prior weight must be positive.'
        print('Using latent prior (%s): weight = %.03f' % (latprior, w_latprior))
    
    replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device
    )

    agent = make_agent(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        action_max=action_max,
        device=device,
        args=args
    )

    L = Logger(args.work_dir)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    for step in range(1, args.num_train_steps + 1):
        # evaluate the agent periodically
        if step % args.eval_freq == 0:
            print('Running evaluation: step', step)
            L.log('eval/step', step)
            L.log('eval/episode', episode)
            if args.seed is not None: L.log('eval/seed', args.seed)
            L.log('eval/noisy_dims', args.noisy_dims)
            evaluate(eval_env, agent, args.num_eval_episodes, L)
            if args.save_model:
                agent.save(model_dir, step)
            if args.save_buffer:
                replay_buffer.save(buffer_dir)

        if done:
            if step > 2 * args.init_steps:
                L.log('train/step', step)
                L.log('train/episode', episode)
                L.log('train/episode_reward', episode_reward)
                if args.seed is not None: L.log('train/seed', args.seed)
                L.dump()
            
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            reward = 0

        # sample action for data collection
        if step <= args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.select_action(obs, torch_mode=False)
        
        # run training update
        if step > args.init_steps:
            agent.update(replay_buffer, L, step)

        curr_reward = reward
        next_obs, reward, done, _ = env.step(action)

        if use_IR:
            intrinsic_reward = agent.compute_curiosity_reward(obs, next_obs, action, IR_type, IR_max)
            if step % 1000 == 0:
                print(step, 'IR: %.04f, ER: %.04f' % (intrinsic_reward, reward))
            L.log('train/ir', intrinsic_reward)
            reward += intrinsic_reward
        
        # allow infinite bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
        episode_reward += reward
        # curr_reward: previous iteration reward
        # reward: current iteration reward after taking the current action
        replay_buffer.add(obs, action, curr_reward, reward, next_obs, done_bool)
        np.copyto(replay_buffer.k_obses[replay_buffer.idx - args.k], next_obs)
        
        obs = next_obs
        episode_step += 1
    print('Training completed in %.02f mins' % ((time.time() - start_time) / 60))

if __name__ == "__main__":
    main()
