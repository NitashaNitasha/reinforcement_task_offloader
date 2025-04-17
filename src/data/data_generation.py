import numpy as np
import pandas as pd
import os
import logging
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

def generate_synthetic_dataset(n_samples=5000, output_path='data/processed/synthetic_dataset.csv'):
    """
    Generate synthetic dataset for edge computing offloading
    
    Args:
        n_samples (int): Number of samples to generate
        output_path (str): Path to save the generated dataset
        
    Returns:
        df (pd.DataFrame): Generated dataset
    """
    logger.info(f"Generating synthetic dataset with {n_samples} samples")
    
    # Create directory if it doesn't exist and if there is a directory component
    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    # Random seed for reproducibility
    np.random.seed(42)
    
    # Generate task characteristics
    data_size = np.random.lognormal(mean=2.0, sigma=1.2, size=n_samples) * 0.1  # in MB
    cpu_cycles = np.random.lognormal(mean=8.0, sigma=1.5, size=n_samples) * 1e6  # in cycles
    memory_required = np.random.lognormal(mean=1.5, sigma=1.0, size=n_samples) * 50  # in MB
    
    # Generate deadlines based on task complexity
    # More complex tasks (larger CPU cycles) have longer deadlines
    deadline_base = cpu_cycles / 1e7  # Base deadline in seconds
    deadline_variation = np.random.uniform(0.8, 1.5, size=n_samples)  # Variation factor
    deadline = deadline_base * deadline_variation  # Final deadline in seconds
    
    # IoT device characteristics
    iot_cpu_power = np.random.uniform(0.5, 2.0, size=n_samples) * 1e9  # in cycles/second
    iot_battery = np.random.uniform(50, 100, size=n_samples)  # in percentage
    
    # Edge server characteristics
    efaas_core_power = np.random.uniform(3.0, 10.0, size=n_samples) * 1e9  # in cycles/second
    efaas_load = np.random.uniform(0.1, 0.8, size=n_samples)  # load factor
    
    # Network characteristics
    bandwidth = np.random.uniform(1.0, 10.0, size=n_samples)  # in Mbps
    network_delay = np.random.uniform(0.01, 0.1, size=n_samples)  # in seconds
    channel_gain = np.random.uniform(0.5, 1.0, size=n_samples)  # channel gain
    noise = np.random.uniform(0.01, 0.05, size=n_samples)  # noise level
    
    # Create dataframe
    df = pd.DataFrame({
        'task_id': range(n_samples),
        'data_size': data_size,
        'cpu_cycles': cpu_cycles,
        'memory_required': memory_required,
        'deadline': deadline,
        'iot_cpu_power': iot_cpu_power,
        'iot_battery': iot_battery,
        'efaas_core_power': efaas_core_power,
        'efaas_load': efaas_load,
        'bandwidth': bandwidth,
        'network_delay': network_delay,
        'channel_gain': channel_gain,
        'noise': noise
    })
    
    # Calculate local execution metrics
    df['local_execution_time'] = df['cpu_cycles'] / df['iot_cpu_power']
    df['local_energy'] = 0.5 * df['iot_cpu_power'] * df['local_execution_time'] * 1e-9  # in Joules
    
    # Calculate transmission metrics
    df['transmission_time'] = (df['data_size'] * 8) / (df['bandwidth'])  # in seconds
    
    # Calculate edge execution metrics
    df['edge_execution_time'] = df['cpu_cycles'] / df['efaas_core_power']
    df['total_offload_time'] = df['transmission_time'] + df['edge_execution_time'] + df['network_delay']
    df['offload_energy'] = 0.5 * df['transmission_time']  # in Joules
    
    # Determine optimal execution location (0=local, 1=offload)
    # Simplified decision: offload if it's faster and deadline is tight
    df['optimal_action'] = ((df['total_offload_time'] < df['local_execution_time']) & 
                           (df['local_execution_time'] > 0.8 * df['deadline'])).astype(int)
    
    # Add some randomness to optimal action to prevent overfit
    random_flip = np.random.uniform(0, 1, size=n_samples) < 0.1  # Flip 10% of decisions randomly
    df.loc[random_flip, 'optimal_action'] = 1 - df.loc[random_flip, 'optimal_action']
    
    # Save to CSV
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save dataset: {e}")
    
    return df


def analyze_dataset(dataset_path='data/processed/synthetic_dataset.csv'):
    """
    Analyze dataset and print statistics
    
    Args:
        dataset_path (str): Path to the dataset
    """
    try:
        df = pd.read_csv(dataset_path)
        
        print(f"Dataset shape: {df.shape}")
        print("\nFeature statistics:")
        print(df.describe().round(2))
        
        # Calculate additional statistics
        local_faster = (df['local_execution_time'] < df['total_offload_time']).mean() * 100
        offload_energy_saved = (df['local_energy'] > df['offload_energy']).mean() * 100
        deadline_met_local = (df['local_execution_time'] <= df['deadline']).mean() * 100
        deadline_met_offload = (df['total_offload_time'] <= df['deadline']).mean() * 100
        
        print(f"\nLocal execution faster: {local_faster:.1f}%")
        print(f"Offloading saves energy: {offload_energy_saved:.1f}%")
        print(f"Deadline met with local execution: {deadline_met_local:.1f}%")
        print(f"Deadline met with offloading: {deadline_met_offload:.1f}%")
        print(f"Optimal offloading ratio: {df['optimal_action'].mean() * 100:.1f}%")
        
    except Exception as e:
        logger.error(f"Error analyzing dataset: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Generate dataset
    df = generate_synthetic_dataset(n_samples=5000)
    # Analyze dataset
    analyze_dataset()
