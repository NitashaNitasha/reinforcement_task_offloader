#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the serverless edge computing offloading project.
Handles command-line interface and program execution.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

# Add the src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_generation import generate_synthetic_dataset
from src.agents.mappo_agent import MAPPOAgent
from src.environment.edge_environment import EdgeComputingEnvironment
from src.utils.redis_cache import RedisTaskCache, start_redis_server
from src.utils.performance_metrics import (log_metrics_to_csv,estimate_offloading_cost,compute_forward_pass_time,count_parameters)



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path="experiments/default_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_argparse():
    """Set up command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="DRL-based serverless edge computing offloading system"
    )

    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    subparsers.required = True

    # Generate dataset command
    generate_parser = subparsers.add_parser('generate', help='Generate synthetic dataset')
    generate_parser.add_argument('--samples', type=int, default=5000,
                              help='Number of samples to generate')
    generate_parser.add_argument('--output', type=str, default='data/synthetic_dataset.csv',
                              help='Output path for the dataset')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the MAPPO agent')
    train_parser.add_argument('--dataset', type=str, default='data/synthetic_dataset.csv',
                           help='Path to the dataset')
    train_parser.add_argument('--config', type=str, default='experiments/default_config.yaml',
                           help='Path to the configuration file')
    train_parser.add_argument('--use-cache', action='store_true',
                           help='Use Redis cache during training')
    train_parser.add_argument('--episodes', type=int, default=100,
                           help='Number of episodes for training')
    train_parser.add_argument('--steps', type=int, default=1000,
                           help='Number of steps per episode')

    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    evaluate_parser.add_argument('--model', type=str, required=True,
                               help='Path to the trained model')
    evaluate_parser.add_argument('--dataset', type=str, default='data/processed/synthetic_dataset.csv',
                               help='Path to the dataset')
    evaluate_parser.add_argument('--use-cache', action='store_true',
                               help='Use Redis cache during evaluation')
    evaluate_parser.add_argument('--episodes', type=int, default=20,
                               help='Number of episodes for evaluation')

    # Visualize command
    visualize_parser = subparsers.add_parser('visualize', help='Visualize training or evaluation results')
    visualize_parser.add_argument('--data', type=str, required=True,
                                help='Path to the data file containing results')
    visualize_parser.add_argument('--output', type=str, default='results/figures',
                                help='Output directory for figures')

    # Redis command
    redis_parser = subparsers.add_parser('redis', help='Start Redis server')

    return parser


def generate_mode(args):
    """Generate synthetic dataset."""
    logger.info(f"Generating synthetic dataset with {args.samples} samples...")
    output_path = args.output
    
    # Log the output path for debugging
    logger.info(f"Output path: {output_path}")
    
    # Ensure output directory exists if there is a directory component
    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    generate_synthetic_dataset(n_samples=args.samples, output_path=output_path)
    logger.info(f"Dataset generated and saved to {output_path}")


def train_mode(args):
    """Train the DRL agent (MAPPO or DQN)."""
    import time
    from datetime import datetime

    logger.info("Starting training mode...")
    start_time = time.time()

    # Load configuration
    config = load_config(args.config)

    # Override config with command-line arguments
    config['environment']['dataset_path'] = args.dataset
    config['environment']['use_cache'] = args.use_cache
    config['training']['epochs'] = args.episodes
    config['environment']['simulation_steps'] = args.steps

    # Create environment
    env = EdgeComputingEnvironment(
        dataset_path=config['environment']['dataset_path'],
        use_cache=config['environment']['use_cache'],
        simulation_steps=config['environment']['simulation_steps'],
        batch_size=config['environment']['batch_size']
    )

    # Determine agent type
    agent_type = config.get("agent_type", "MAPPO").upper()

    if agent_type == "MAPPO":
        logger.info("Using MAPPO agent")
        agent = MAPPOAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            n_agents=config['agent']['n_agents'],
            learning_rate=config['agent']['learning_rate'],
            hidden_dim=config['agent']['hidden_dim'],
            lstm_dim=config['agent']['lstm_dim']
        )
        from src.train_offloading_agent import train_agent_with_per_epoch_metrics
        train_fn = train_agent_with_per_epoch_metrics

    elif agent_type == "DQN":
        logger.info("Using DQN agent")
        from src.agents.dqn_agent import DQNAgent
        from src.train_offloading_agent import train_dqn_with_per_epoch_metrics
        agent = DQNAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            learning_rate=config['agent']['learning_rate'],
            gamma=config['agent'].get('gamma', 0.99),
            epsilon_start=config['agent'].get('epsilon_start', 1.0),
            epsilon_min=config['agent'].get('epsilon_min', 0.05),
            epsilon_decay=config['agent'].get('epsilon_decay', 0.995),
            buffer_size=config['agent'].get('buffer_size', 10000),
            # batch_size=config['agent'].get('batch_size', 64),
            batch_size=128
        )
        train_fn = train_dqn_with_per_epoch_metrics

    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")

    # Get observation space dimension for inference time calculation
    if hasattr(env.observation_space, 'shape'):
        obs_dim = env.observation_space.shape[0]
    else:
        # For Dict observation spaces, estimate total dimension
        obs_dim = sum(space.shape[0] for space in env.observation_space.values())

    # Calculate policy network parameters once
    policy_net = agent.policy_net if hasattr(agent, 'policy_net') else agent.actor if hasattr(agent, 'actor') else None
    if policy_net is not None:
        n_params = count_parameters(policy_net)
        inference_time = compute_forward_pass_time(policy_net, input_shape=(1, obs_dim))
    else:
        logger.warning("Could not identify policy network for metrics calculation")
        n_params = 0
        inference_time = 0

    # Prepare for metrics collection
    avg_task_compute_time = 0.05  # Example value in seconds
    serverless_cost_per_second = 0.00001667  # Example value in dollars
    avg_data_size_MB = 2.0  # Example value in MB
    communication_cost_per_MB = 0.00001  # Example value in dollars
    cost = estimate_offloading_cost(
        t_compute=avg_task_compute_time,
        lambda_cost=serverless_cost_per_second,
        performance=1.0,  # Normalized performance
        data_size_MB=avg_data_size_MB,
        comm_cost_per_MB=communication_cost_per_MB
    )

    # Get timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = f"results/metrics_{agent_type.lower()}_{timestamp}.csv"

    # Initialize metrics dict with some constant values
    base_metrics = {
        "agent_type": agent_type,
        "params": n_params,
        "inference_time": inference_time,
        "estimated_cost": cost,
        "steps_per_episode": config['environment']['simulation_steps'],
        "n_agents": config['agent']['n_agents'] if agent_type == "MAPPO" else 1
    }

    # Train agent with per-epoch metrics
    logger.info(f"Training agent for {args.episodes} episodes with {args.steps} steps per episode...")
    epoch_metrics = train_fn(agent, env, config, base_metrics, metrics_file)

    total_training_time = time.time() - start_time
    logger.info(f"Training completed in {total_training_time:.2f} seconds")
    logger.info(f"Performance metrics saved to {metrics_file}")

    # Save the trained model with timestamp
    model_path = f"models/{agent_type.lower()}_model_{timestamp}.pt"
    if hasattr(agent, 'save'):
        agent.save(model_path)
        logger.info(f"Model saved to {model_path}")
    else:
        logger.warning("Agent does not have a save method. Model not saved.")

    return epoch_metrics



def evaluate_mode(args):
    """Evaluate a trained model."""
    logger.info(f"Evaluating model: {args.model}")
    
    # Load configuration
    config = load_config()
    config['evaluation']['n_episodes'] = args.episodes
    
    # Create environment
    env = EdgeComputingEnvironment(
        dataset_path=args.dataset,
        use_cache=args.use_cache
    )
    
    # Load agent
    agent = MAPPOAgent.load_from_checkpoint(args.model)
    
    # Evaluate agent
    from src.train_offloading_agent import evaluate_agent
    results = evaluate_agent(agent, env, config)
    
    logger.info(f"Evaluation results: {results}")




def redis_mode(args):
    """Start Redis server."""
    logger.info("Starting Redis server...")
    start_redis_server()


def main():
    """Main entry point."""
    parser = setup_argparse()
    args = parser.parse_args()

    
    # Execute the selected mode
    if args.mode == 'generate':
        generate_mode(args)
    elif args.mode == 'train':
        train_mode(args)
    elif args.mode == 'evaluate':
        evaluate_mode(args)
    elif args.mode == 'redis':
        redis_mode(args)


if __name__ == "__main__":
    main()