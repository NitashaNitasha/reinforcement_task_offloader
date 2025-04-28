
import time
import csv
import os
import numpy as np
import torch


def count_parameters(model):
    """
    Count the number of trainable parameters in a PyTorch model.

    Args:
        model: PyTorch model

    Returns:
        int: Number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_forward_pass_time(model, input_shape, num_runs=1000):
    """
    Measure the average time for a forward pass through the model.

    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (batch_size, features)
        num_runs: Number of forward passes to average over

    Returns:
        float: Average inference time in milliseconds
    """
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_shape, device=device)

    # Warm-up run
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Timed runs
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    end_time = time.time()

    avg_time_ms = (end_time - start_time) * 1000 / num_runs
    return avg_time_ms


def estimate_offloading_cost(t_compute, lambda_cost, performance=1.0,
                             data_size_MB=0.0, comm_cost_per_MB=0.0):
    """
    Estimate the cost of offloading a task to serverless infrastructure.

    Args:
        t_compute: Computation time in seconds
        lambda_cost: Cost per second of compute (e.g., AWS Lambda cost)
        performance: Performance factor (higher means faster execution, default=1.0)
        data_size_MB: Size of data to transfer in MB
        comm_cost_per_MB: Communication cost per MB

    Returns:
        float: Estimated cost in dollars
    """
    compute_cost = (t_compute / performance) * lambda_cost
    comm_cost = data_size_MB * comm_cost_per_MB

    total_cost = compute_cost + comm_cost
    return total_cost


def calculate_mappo_complexity(n_agents, policy_hidden_dim, critic_hidden_dim, obs_dim, action_dim):
    """
    Calculate theoretical time complexity for MAPPO operations.

    Args:
        n_agents: Number of agents
        policy_hidden_dim: Hidden dimension of policy network
        critic_hidden_dim: Hidden dimension of critic network
        obs_dim: Observation dimension
        action_dim: Action dimension

    Returns:
        dict: Complexity estimates
    """
    # Policy network complexity (per agent)
    policy_complexity = obs_dim * policy_hidden_dim + policy_hidden_dim * action_dim

    # Centralized critic complexity
    critic_complexity = (obs_dim * n_agents) * critic_hidden_dim + critic_hidden_dim * 1

    # Combined complexity
    total_complexity = n_agents * policy_complexity + critic_complexity

    return {
        "policy_per_agent": policy_complexity,
        "critic": critic_complexity,
        "total": total_complexity
    }


def calculate_dqn_complexity(obs_dim, action_dim, hidden_dim, batch_size):
    """
    Calculate theoretical time complexity for DQN operations.

    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension
        hidden_dim: Hidden dimension of Q-network
        batch_size: Batch size for training

    Returns:
        dict: Complexity estimates
    """
    # Q-network complexity
    q_network_complexity = obs_dim * hidden_dim + hidden_dim * action_dim

    # Per-batch complexity
    batch_complexity = batch_size * q_network_complexity

    return {
        "q_network": q_network_complexity,
        "per_batch": batch_complexity
    }

def calculate_costs(data,n_agents):
    computation_costs = []
    transmission_costs = []
    delays = []

    price_per_ghz_s = 0.001
    price_per_kb = 0.00005

    for i in range(len(data['CPU_Cycles'])):
        cpu_cycles = float(data['CPU_Cycles'][i])
        min_cpu_speed = float(data['Min_CPU_Speed'][i])
        exec_time = float(data['ExecutionTime_s'][i])
        data_size = float(data['DataSize_kB'][i])
        total_exec_time = float(data['TotalExecutionTime_s'][i])
        deadline = float(data['Deadline_s'][i])

        # Avoid division by zero
        if min_cpu_speed > 0:
            computation_cost = (cpu_cycles / min_cpu_speed) * price_per_ghz_s
        else:
            computation_cost = 0.0

        transmission_cost = data_size * price_per_kb
        delay = max(total_exec_time - deadline, 0)

        # 🔍 Debug: print actual values being used
        print(f"[Step {i}] CPU: {cpu_cycles:.4f}, Speed: {min_cpu_speed:.4f}, "
              f"CompCost: {computation_cost:.6f}, DataSize: {data_size:.2f}, "
              f"TransCost: {transmission_cost:.6f}, ExecTime: {total_exec_time:.4f}, "
              f"Deadline: {deadline:.4f}, Delay: {delay:.6f}")

        computation_costs.append(computation_cost)
        transmission_costs.append(transmission_cost)
        delays.append(delay)
    try:
        average_delay = [d/n_agents for d in delays]
    except ZeroDivisionError:
        return delays

    return computation_costs, transmission_costs, average_delay



def log_metrics_to_csv(metrics_dict, output_path="results/performance_metrics.csv"):
    """
    Log performance metrics to a CSV file.

    Args:
        metrics_dict: Dictionary of metrics to log
        output_path: Path to CSV file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Check if file exists to determine if header is needed
    file_exists = os.path.isfile(output_path)

    with open(output_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=metrics_dict.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(metrics_dict)


def calculate_theoretical_flops(model, input_shape):
    """
    Estimate theoretical FLOPs for a PyTorch model.
    This is a simplified estimation and may not be accurate for all model types.
    For more accurate measurement, consider using libraries like fvcore or ptflops.

    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (batch_size, features)

    Returns:
        int: Estimated FLOPs
    """
    total_flops = 0

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # For each linear layer: FLOPs = 2 * in_features * out_features
            # (multiplication and addition for each weight)
            flops = 2 * module.in_features * module.out_features
            total_flops += flops

        elif isinstance(module, torch.nn.Conv2d):
            # For each conv layer: FLOPs = 2 * kernel_size^2 * in_channels * out_channels * output_size^2
            output_height = (input_shape[2] - module.kernel_size[0] + 2 * module.padding[0]) // module.stride[0] + 1
            output_width = (input_shape[3] - module.kernel_size[1] + 2 * module.padding[1]) // module.stride[1] + 1

            flops = 2 * module.kernel_size[0] * module.kernel_size[
                1] * module.in_channels * module.out_channels * output_height * output_width
            total_flops += flops

        elif isinstance(module, torch.nn.LSTM):
            # Simplified LSTM FLOPs calculation
            input_size = module.input_size
            hidden_size = module.hidden_size
            num_layers = module.num_layers

            # Each gate in LSTM has weights for input and hidden state
            gate_params = 4 * hidden_size * (input_size + hidden_size)

            # For each time step and layer
            flops = input_shape[1] * num_layers * gate_params * 2  # *2 for both multiply and add
            total_flops += flops

    return total_flops