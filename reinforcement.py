import wandb
import torch

from agents import DRQN
from environments import TMaze
from memory import ReplayBuffer

from models_pytorch.utils import set_seed
from argparse import ArgumentParser


CELLS = ('lstm', 'chrono', 'gru', 'mgu', 'brc', 'nbrc')


def main(config):

    torch.set_num_threads(4)

    # Set random seed for reproducibility
    args.seed = set_seed(args.seed)

    # Entity can be changed
    tags = [args.wandb_tag] if args.wandb_tag else None
    run = wandb.init(project="warmup-reinforcement", config=config, tags=tags)

    # Environment
    env = TMaze(config.length)
    observation_size = env.observation_size
    num_actions = env.num_actions

    # Agent and Q-networks
    network_args = {
        'cell': config.cell,
        'input_size': observation_size + 1 + num_actions,
        'hidden_size': config.hidden_size,
        'num_layers': config.num_layers,
        'output_size': num_actions,
        'regression': True}

    if config.cell == 'chrono':
        if config.t_max is None:
            network_args['t_max'] = 3 * config.length
        else:
            network_args['t_max'] = config.t_max

    agent = DRQN(num_actions, observation_size, **network_args)
    buffer = ReplayBuffer(config.buffer_capacity)

    # Filling replay buffer with exploration trajectories
    while not buffer.is_full:
        agent.play(env, epsilon=1.0, buffer=buffer)

    # Warmup
    if config.warmup:
        agent.Q.warmup(buffer.get_input_sequences(), config.num_steps_warmup,
                       config.learning_rate_warmup, config.batch_size_warmup)

    # Train
    total_loss = float('nan')
    total_frames = 0
    episode_optimality = None
    for episode in range(config.num_episodes + 1):

        if episode % config.evaluation_period == 0:
            stats = agent.eval(env, config.num_rollouts)

            is_optimal = agent.eval_tmaze_optimal(env)
            if is_optimal and episode_optimality is None:
                episode_optimality = episode
            stats['train/optimal'] = int(is_optimal)

            print(f'Episode {episode:04d}')
            print(f'\tMean reward: {stats["train/mean"]: .4f}')

            stats['train/episode'] = episode
            stats['train/loss'] = total_loss / config.evaluation_period

            inputs = buffer.get_input_sequences()

            stats['train/vaa_star'] = agent.Q.mean_vaa_star(
                inputs, config.stabilization_vaa, epsilon=1e-4,
                num_samples=200)

            stats['train/vaa'] = agent.Q.vaa(
                inputs, config.stabilization_vaa, epsilon=1e-4,
                num_samples=200)

            total_loss = 0.0
            wandb.log(stats)

        if episode % config.measure_period == 0:
            inputs = buffer.get_input_sequences()
            vaa_measure = agent.Q.vaa(inputs, config.stabilization_measure,
                                      epsilon=1e-4, num_samples=200)
            wandb.log({
                'train/episode': episode,
                'train/vaa_measure': vaa_measure})

            agent.save(run.id, episode=episode)

        if episode % config.target_update_period == 0:
            agent.update_target()

        trajectory = agent.play(env, epsilon=config.epsilon, buffer=buffer)
        total_frames += trajectory.num_transitions

        for _ in range(config.num_steps_by_episode):

            trajectories = buffer.sample(config.batch_size)
            loss = agent.optimize(trajectories, env.gamma,
                                  config.learning_rate)
            total_loss += loss

    run.summary['episode_optimality'] = episode_optimality


if __name__ == '__main__':

    parser_rl = ArgumentParser(description='Launch RL experiment')
    parser_rl.add_argument("--seed", type=int, default=None)

    # Architecture
    parser_rl.add_argument('--cell', type=str, choices=CELLS, default='gru')
    parser_rl.add_argument('--t-max', type=int, default=None)
    parser_rl.add_argument('--hidden-size', type=int, default=32)
    parser_rl.add_argument('--num-layers', type=int, default=2)

    # Training
    parser_rl.add_argument('--num-episodes', type=int, default=2000)
    parser_rl.add_argument('--batch-size', type=int, default=32)
    parser_rl.add_argument('--buffer-capacity', type=int, default=8192)
    parser_rl.add_argument('--learning-rate', type=float, default=1e-3)
    parser_rl.add_argument('--num-steps-by-episode', type=int, default=10)
    parser_rl.add_argument('--target-update-period', type=int, default=20)
    parser_rl.add_argument('--epsilon', type=float, default=0.2)

    # Evaluation
    parser_rl.add_argument('--evaluation-period', type=int, default=20)
    parser_rl.add_argument('--num-rollouts', type=int, default=50)

    # Warmup
    parser_rl.add_argument('--warmup', action='store_true')
    parser_rl.add_argument('--double', action='store_true')
    parser_rl.add_argument('--num-steps-warmup', type=int, default=100)
    parser_rl.add_argument('--learning-rate-warmup', type=float, default=1e-2)
    parser_rl.add_argument('--batch-size-warmup', type=int, default=32)
    parser_rl.add_argument('--stabilization-vaa', type=int, default=200)
    parser_rl.add_argument('--stabilization-measure', type=int, default=10_000)
    parser_rl.add_argument('--measure-period', type=int, default=200)

    # Environment
    parser_rl.add_argument('--length', type=int, default=10)

    # Wandb
    parser_rl.add_argument('--wandb-tag', type=str, default=None)

    # Parse arguments
    args = parser_rl.parse_args()
    print('\n'.join(f'\033[90m{k}=\033[0m{v}' for k, v in vars(args).items()))

    main(config=args)
