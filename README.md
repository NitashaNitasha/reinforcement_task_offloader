# Serverless Edge Computing Function Offloading with DRL

This project implements a serverless edge computing function offloading system using Multi-Agent Proximal Policy Optimization (MAPPO) reinforcement learning with Redis-based caching.

## System Architecture

The system consists of the following components:

1. **Edge Computing Environment**: Simulates IoT devices and edge servers with dynamic network conditions and resource constraints.
2. **MAPPO Agent**: Deep Reinforcement Learning agent that learns optimal task offloading decisions.
3. **Redis Cache**: Provides caching for offloading decisions to improve system performance.
4. **Data Generation**: Tools to create synthetic datasets for training and evaluation.
5. **Visualization**: Tools for analyzing and plotting training/evaluation metrics.

## Key Features

- **Intelligent Function Offloading**: Uses DRL to make optimal offloading decisions based on task characteristics, device state, edge server state, and network conditions.
- **Redis Caching**: Improves performance by caching offloading decisions for similar tasks.
- **Multi-Agent Support**: Supports multiple agents for distributed scenarios.
- **Comprehensive Metrics**: Tracks latency, energy consumption, deadline violations, and cache hit ratio.
- **Visualization Tools**: Provides detailed graphs for training and evaluation results.

## Installation

### Prerequisites

- Python 3.7+
- PyTorch
- Redis

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/NitashaNitasha/reinforcement_task_offloader.git
   cd reinforcement_task_offloader
   ```

2. Install dependencies:
   ```
   python -m venv venv
   ./venv/Scripts/activate
   pip install -r requirements.txt
   ```

3. Start Redis server (optional - for caching):
   ```
   python main.py redis
   ```

## Usage


### 1. Train the MAPPO agent

```bash
 python main.py train --config experiments/mappo_config.yaml --episodes 1000 --steps 1000 --use-cache

```
### 2. Train the DQN agent

```bash
python main.py train --config experiments/dqn_config.yaml --episodes 1000 --steps 1000 --use-cache
```
### 3. Train the PPO agent

```bash
 python main.py train --config experiments/ppo_config.yaml --episodes 1000 --steps 1000 --use-cache

```
### 4. Evaluate a trained model

```bash
python main.py evaluate --model models/agent_0_final.pth --dataset synthetic_dataset.csv --use-cache
```

### 5. Visualize results

```bash
python visualization.py
```

## System Components

### 1. Edge Environment (`edge_environment.py`)

- Simulates serverless edge computing environment
- Implements task queues and processing logic
- Handles task offloading decisions
- Tracks system metrics

### 2. MAPPO Agent (`mappo_agent.py`)

- Implements the Multi-Agent Proximal Policy Optimization algorithm
- Neural network architecture with LSTM for temporal dependencies
- Policy and value heads for actor-critic architecture
- Experience replay and GAE for stable training

### 3. Redis Cache (`redis_cache.py`)

- Provides caching for offloading decisions
- Stores and retrieves task execution results
- Falls back to simulated caching when Redis is unavailable

### 4. Training & Evaluation (`train_offloading_agent.py`)

- Training loop for the MAPPO agent
- Evaluation routines for trained models
- Metrics collection and visualization

## Command Line Interface

The system provides a comprehensive CLI via `main.py`:

```
python main.py [mode] [options]
```

Available modes:
- `generate`: Generate synthetic dataset
- `train`: Train the MAPPO agent
- `evaluate`: Evaluate a trained model
- `visualize`: Visualize training or evaluation results
- `redis`: Run a Redis server for caching

For detailed options:

```
python main.py [mode] --help
```

## MDP Formulation

1. **State Space**:
   - Task characteristics (data size, CPU cycles, memory, deadline)
   - IoT device state (computation capacity, queue latency)
   - Edge server state (computation capacity, queue latency)
   - Network state (transmission rate, transmission power, channel gain, SINR)

2. **Action Space**:
   - Binary decision: Execute locally (0) or offload to edge (1)

3. **Reward Function**:
   - Negative of latency cost if deadline is met
   - Penalty if deadline is violated

```
