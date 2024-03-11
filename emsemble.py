import os
import argparse
import sys
import random
sys.path.append('/Users/aniket/github/229PokerProphet-1/rlcard')
import torch

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)


def train(args):
    device = get_device()
    set_seed(args.seed)
    env = rlcard.make(
        args.env,
        config={
            'seed': args.seed,
        }
    )

    ensemble = []
    for _ in range(args.num_agents):
        if args.algorithm == 'dqn':
            from rlcard.agents import DQNAgent
            if args.load_checkpoint_path != "":
                agent = DQNAgent.from_checkpoint(checkpoint=torch.load(args.load_checkpoint_path))
            else:
                agent = DQNAgent(
                    num_actions=env.num_actions,
                    state_shape=env.state_shape[0],
                    mlp_layers=[64,64],
                    device=device,
                    save_path=args.log_dir,
                    save_every=args.save_every
                )
        elif args.algorithm == 'nfsp':
            from rlcard.agents import NFSPAgent
            if args.load_checkpoint_path != "":
                agent = NFSPAgent.from_checkpoint(checkpoint=torch.load(args.load_checkpoint_path))
            else:
                agent = NFSPAgent(
                    num_actions=env.num_actions,
                    state_shape=env.state_shape[0],
                    hidden_layers_sizes=[64,64],
                    q_mlp_layers=[64,64],
                    device=device,
                    save_path=args.log_dir,
                    save_every=args.save_every
                )
        ensemble.append(agent)

    opponents = [RandomAgent(num_actions=env.num_actions) for _ in range(env.num_players - 1)]
    env.set_agents(ensemble) 

    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            #agent = random.choice(ensemble) 
            for agent in ensemble:
                trajectories, payoffs = env.run(is_training=True)
                trajectories = reorganize(trajectories, payoffs)
                for ts in trajectories[0]:
                    agent.feed(ts)

                if episode % args.evaluate_every == 0:
                    logger.log_performance(
                        episode,
                        tournament(
                            env, 
                            args.num_eval_games, 
                            # [agent],  # Use the current agent for evaluation
                            # opponents
                        )[0]
                    )

        csv_path, fig_path = logger.csv_path, logger.fig_path

    plot_curve(csv_path, fig_path, args.algorithm)

    for i, agent in enumerate(ensemble):
        save_path = os.path.join(args.log_dir, f'model_{i}.pth')
        torch.save(agent, save_path)
        print(f'Model {i} saved in {save_path}')


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser("DQN/NFSP example in RLCard")
    parser.add_argument('--env', type=str, default='leduc-holdem', choices=['blackjack', 'leduc-holdem', 'limit-holdem', 'doudizhu', 'mahjong', 'no-limit-holdem', 'uno', 'gin-rummy', 'bridge'])
    parser.add_argument('--algorithm', type=str, default='dqn', choices=['dqn', 'nfsp'])
    parser.add_argument('--cuda', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_episodes', type=int, default=5000)
    parser.add_argument('--num_eval_games', type=int, default=2000)
    parser.add_argument('--evaluate_every', type=int, default=100)
    parser.add_argument('--log_dir', type=str, default='experiments/leduc_holdem_dqn_result/')
    parser.add_argument(
        "--load_checkpoint_path", type=str, default="")
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--save_every", type=int, default=-1)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    # Train the ensemble
    train(args)
