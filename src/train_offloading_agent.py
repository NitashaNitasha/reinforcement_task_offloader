import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from datetime import datetime
import yaml
from tensorboard import program
import csv
import os
from collections import defaultdict
from src.utils.performance_metrics import calculate_costs
import threading
from src.utils.performance_metrics import log_metrics_to_csv
from src.agents.mappo_agent import MAPPOAgent
from src.agents.dqn_agent import DQNAgent
from src.environment.edge_environment import EdgeComputingEnvironment

logger = logging.getLogger(__name__)



def evaluate_agent(agent, env, config, n_episodes=20, verbose=True):
    """
    Evaluate a trained agent
    
    Args:
        agent (MAPPOAgent): Agent to evaluate
        env (EdgeComputingEnvironment): Evaluation environment
        config (dict): Configuration dictionary
        n_episodes (int): Number of episodes to evaluate
        verbose (bool): Whether to print evaluation progress
        
    Returns:
        metrics (dict): Evaluation metrics
    """
    # Turn on evaluation mode
    deterministic = config.get('evaluation', {}).get('deterministic', True)
    
    # Reset metrics
    rewards = []
    latencies = []
    deadline_violations = []
    
    # Evaluate for n_episodes
    if verbose:
        logger.info(f"Evaluating agent for {n_episodes} episodes...")
    
    for episode in range(1, n_episodes + 1):
        episode_rewards = []
        episode_latencies = []
        episode_deadline_violations = []
        
        # Reset environment
        states = env.reset()
        agent.reset_lstm_states()
        
        # Run episode
        done = False
        while not done:
            # Select actions
            actions, _, _ = agent.act(states)
            
            # Execute actions
            next_states, episode_reward, dones, info = env.step(actions)
            
            # Update states
            states = next_states
            done = any(dones)
            
            # Collect statistics
            episode_rewards.extend(episode_reward)
            episode_latencies.extend(info['latencies'])
            episode_deadline_violations.extend(info['deadline_violations'])
        
        # Record episode statistics
        rewards.append(np.mean(episode_rewards))
        latencies.append(np.mean(episode_latencies))
        deadline_violations.append(np.mean(episode_deadline_violations) * 100)  # As percentage
        
        if verbose and episode % 5 == 0:
            logger.info(f"Episode {episode}/{n_episodes} - " +
                       f"Reward: {rewards[-1]:.2f}, " +
                       f"Latency: {latencies[-1]:.4f}, " +
                       f"Deadline Violations: {deadline_violations[-1]:.1f}%")
    
    # Get environment metrics
    env_metrics = env.get_metrics()
    offload_ratio = env_metrics['offload_ratio'] * 100
    cache_hit_ratio = env_metrics['cache_hit_ratio'] * 100
    
    # Calculate statistics
    metrics = {
        'reward_mean': np.mean(rewards),
        'reward_std': np.std(rewards),
        'latency_mean': np.mean(latencies),
        'latency_std': np.std(latencies),
        'deadline_violations_mean': np.mean(deadline_violations),
        'offload_ratio': offload_ratio,
        'cache_hit_ratio': cache_hit_ratio
    }
    
    if verbose:
        logger.info("Evaluation Results:")
        logger.info(f"Reward: {metrics['reward_mean']:.2f} ± {metrics['reward_std']:.2f}")
        logger.info(f"Latency: {metrics['latency_mean']:.4f} ± {metrics['latency_std']:.4f}")
        logger.info(f"Deadline Violations: {metrics['deadline_violations_mean']:.1f}%")
        logger.info(f"Offload Ratio: {metrics['offload_ratio']:.1f}%")
        logger.info(f"Cache Hit Ratio: {metrics['cache_hit_ratio']:.1f}%")
    
    return metrics



def train_agent_with_per_epoch_metrics(agent, env, config, base_metrics, metrics_file=None):
    """
    Train an agent with per-epoch metrics logging including computation, transmission, and delay penalties.

    Args:
        agent: The reinforcement learning agent to train
        env: The environment to train in
        config: Training configuration
        base_metrics: Base metrics to include in every epoch
        metrics_file: File to save metrics to

    Returns:
        dict: Metrics for the final epoch
    """
    n_episodes = config.get('episodes', 1000)
    max_steps_per_episode = config.get('steps', 1000)

    all_rewards = []
    avg_rewards = []

    for epoch in range(n_episodes):
        start_time = time.time()
        states = env.reset()
        agent.reset_lstm_states()

        episode_rewards = []
        total_reward = 0
        done = False
        steps = 0

        # Collect cost-related episode data
        episode_data = {
            'CPU_Cycles': [],
            'Min_CPU_Speed': [],
            'ExecutionTime_s': [],
            'DataSize_kB': [],
            'TotalExecutionTime_s': [],
            'Deadline_s': []
        }

        while not done and steps < max_steps_per_episode:
            agent_ids = list(range(agent.n_agents))

            if not agent_ids:
                break

            agent_states = [states[i] for i in agent_ids]
            actions, log_probs, values = agent.act(agent_states, agent_ids)
            next_states, rewards, dones, info = env.step(actions)

            # Store transition
            agent.store_transition(
                agent_ids, agent_states, actions, log_probs, rewards, values, dones
            )

            # Collect cost-relevant info
            if isinstance(info, list):
                for agent_info in info:
                    episode_data['CPU_Cycles'].append(agent_info.get('CPU_Cycles', 0.0))
                    episode_data['Min_CPU_Speed'].append(agent_info.get('Min_CPU_Speed', 0.0))
                    episode_data['ExecutionTime_s'].append(agent_info.get('ExecutionTime_s', 0.0))
                    episode_data['DataSize_kB'].append(agent_info.get('DataSize_kB', 0.0))
                    episode_data['TotalExecutionTime_s'].append(agent_info.get('TotalExecutionTime_s', 0.0))
                    episode_data['Deadline_s'].append(agent_info.get('Deadline_s', 0.0))

            states = next_states
            episode_rewards.extend(rewards)
            total_reward = sum(rewards) / len(rewards)

            done = all(dones) if isinstance(dones, list) else dones
            steps += 1

            if hasattr(agent, 'update_frequency') and steps % agent.update_frequency == 0:
                with torch.no_grad():
                    next_active_agents = env.get_active_agents()
                    next_agent_ids = [i for i in range(len(next_active_agents)) if next_active_agents[i]]
                    next_agent_states = [states[i] for i in next_agent_ids]

                    if next_agent_ids:
                        _, _, next_values = agent.act(next_agent_states, next_agent_ids)
                    else:
                        next_values = [0.0] * len(agent_ids)

                update_info = agent.update(next_values)

        # Final agent update
        next_values = [0.0] * agent.n_agents
        update_info = agent.update(next_values)

        epoch_time = time.time() - start_time

        # Metric dictionary for this epoch
        epoch_metrics = {
            'epoch': epoch,
            'reward': total_reward,
        }

        if update_info and len(update_info) > 0:
            avg_policy_loss = sum(info.get('policy_loss', 0.0) for info in update_info) / len(update_info)
            avg_value_loss = sum(info.get('value_loss', 0.0) for info in update_info) / len(update_info)
            avg_entropy_loss = sum(info.get('entropy_loss', 0.0) for info in update_info) / len(update_info)

            epoch_metrics.update({
                'policy_loss': avg_policy_loss,
                'value_loss': avg_value_loss,
                'entropy_loss': avg_entropy_loss,
                'total_loss': avg_policy_loss + avg_value_loss - avg_entropy_loss,
            })
        else:
            epoch_metrics.update({
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy_loss': 0.0,
                'total_loss': 0.0,
            })

        # Cost calculation
        required_fields = list(episode_data.keys())
        if all(len(episode_data[field]) > 0 for field in required_fields):
            comp_costs, trans_costs, delay_penalties = calculate_costs(episode_data)
            epoch_metrics['computation_cost'] = np.mean(comp_costs)
            epoch_metrics['transmission_cost'] = np.mean(trans_costs)
            epoch_metrics['delay_penalty'] = np.mean(delay_penalties)
        else:
            epoch_metrics['computation_cost'] = 0.0
            epoch_metrics['transmission_cost'] = 0.0
            epoch_metrics['delay_penalty'] = 0.0

        # Finalize metrics
        epoch_metrics['training_time'] = epoch_time
        epoch_metrics.update(base_metrics)

        if update_info:
            for i, agent_info in enumerate(update_info):
                agent_metrics = {
                    f'agent_{i}_policy_loss': agent_info.get('policy_loss', 0.0),
                    f'agent_{i}_value_loss': agent_info.get('value_loss', 0.0),
                    f'agent_{i}_entropy_loss': agent_info.get('entropy_loss', 0.0),
                }
                epoch_metrics.update(agent_metrics)

        all_rewards.append(total_reward)
        avg_rewards.append(sum(all_rewards[-100:]) / min(len(all_rewards), 100))

        if metrics_file:
            log_metrics_to_csv(epoch_metrics, metrics_file)

        if epoch % 10 == 0 or epoch == n_episodes - 1:
            print(f"Episode {epoch}/{n_episodes}, Reward: {total_reward:.2f}, "
                  f"Avg Reward (100): {avg_rewards[-1]:.2f}, Time: {epoch_time:.2f}s")

    return epoch_metrics

def train_dqn_with_per_epoch_metrics(agent, env, config, base_metrics, metrics_file):
    """
    Train a DQN agent with metrics collected for each epoch.

    Args:
        agent: The DQN agent to train
        env: The environment to train in
        config: Configuration dictionary
        base_metrics: Dictionary of metrics that remain constant across epochs
        metrics_file: Path to save the metrics CSV file

    Returns:
        Dictionary containing lists of per-epoch metrics
    """


    # Ensure directory exists
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)

    # Initialize CSV file with headers
    with open(metrics_file, 'w', newline='') as f:
        fieldnames = [
            'epoch', 'reward', 'epsilon', 'loss', 'training_time',
            'agent_type', 'params', 'inference_time', 'estimated_cost',
            'steps_per_episode', 'n_agents',
            'computation_cost', 'transmission_cost', 'delay_penalty'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    n_epochs = config['training']['epochs']
    steps_per_episode = config['environment']['simulation_steps']
    all_metrics = defaultdict(list)

    for epoch in range(1, n_epochs + 1):
        epoch_start_time = time.time()
        state = env.reset()
        done = False
        total_reward = 0
        losses = []
        episode_step = 0

        # Track values for cost calculation
        episode_data = {
            'CPU_Cycles': [],
            'Min_CPU_Speed': [],
            'ExecutionTime_s': [],
            'DataSize_kB': [],
            'TotalExecutionTime_s': [],
            'Deadline_s': []
        }

        while not done and episode_step < steps_per_episode:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step([action])

            # Collect required info for cost metrics
            episode_data['CPU_Cycles'].append(info['CPU_Cycles'][0])
            episode_data['Min_CPU_Speed'].append(info['Min_CPU_Speed'][0])
            episode_data['ExecutionTime_s'].append(info['ExecutionTime_s'][0])
            episode_data['DataSize_kB'].append(info['DataSize_kB'][0])
            episode_data['TotalExecutionTime_s'].append(info['TotalExecutionTime_s'][0])
            episode_data['Deadline_s'].append(info['Deadline_s'][0])

            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward[0]
            episode_step += 1

        epoch_time = time.time() - epoch_start_time
        avg_loss = sum(losses) / len(losses) if losses else 0

        # Base metrics
        epoch_metrics = {
            'epoch': epoch,
            'reward': total_reward,
            'epsilon': agent.epsilon if hasattr(agent, 'epsilon') else 0,
            'loss': avg_loss,
            'training_time': epoch_time,
            **base_metrics
        }

        # Debug: print data before cost calculation
        # print("Raw episode_data before cost calculation:")
        # for k, v in episode_data.items():
        #     print(f"{k}: {v[:3]}")

        # Compute cost metrics if data is valid
        required_fields = list(episode_data.keys())
        if all(len(episode_data[field]) == episode_step for field in required_fields):
            comp_costs, trans_costs, delay_penalties = calculate_costs(episode_data)
            epoch_metrics['computation_cost'] = np.mean(comp_costs)
            epoch_metrics['transmission_cost'] = np.mean(trans_costs)
            epoch_metrics['delay_penalty'] = np.mean(delay_penalties)

        else:
            print("Skipping cost calculation due to missing or mismatched data lengths.")
            epoch_metrics['computation_cost'] = 0.0
            epoch_metrics['transmission_cost'] = 0.0
            epoch_metrics['delay_penalty'] = 0.0

        # # Log to console
        # print(f"[Episode {epoch}] Total Reward: {total_reward:.2f}, "
        #       f"Epsilon: {epoch_metrics['epsilon']:.3f}, Time: {epoch_time:.2f}s")

        # Save to CSV
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(epoch_metrics)

        # Track metrics
        for key, value in epoch_metrics.items():
            all_metrics[key].append(value)

        if hasattr(agent, 'decay_epsilon'):
            agent.decay_epsilon()

    return dict(all_metrics)
