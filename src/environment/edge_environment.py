import numpy as np
import pandas as pd
import gym
from gym import spaces
import os
import logging
from src.utils.redis_cache import RedisTaskCache

logger = logging.getLogger(__name__)

class EdgeComputingEnvironment(gym.Env):
    """
    Edge computing environment for serverless function offloading
    
    Simulates IoT devices and edge servers with task offloading decisions
    """
    
    def __init__(self, dataset_path='data/synthetic_dataset.csv',
                 simulation_steps=1000, batch_size=32, use_cache=False,
                 redis_host='localhost', redis_port=6379):
        """
        Initialize the edge computing environment
        
        Args:
            dataset_path (str): Path to the dataset
            simulation_steps (int): Number of steps in each episode
            batch_size (int): Number of tasks to process in each batch
            use_cache (bool): Whether to use Redis cache
            redis_host (str): Redis host
            redis_port (int): Redis port
        """
        super(EdgeComputingEnvironment, self).__init__()
        
        # Load dataset
        try:
            self.dataset = pd.read_csv(dataset_path)
        except FileNotFoundError:
            # If relative path fails, try absolute path
            module_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(os.path.dirname(module_dir))
            abs_path = os.path.join(project_dir, dataset_path)
            self.dataset = pd.read_csv(abs_path)
            
        # Environment parameters
        self.simulation_steps = simulation_steps
        self.batch_size = batch_size
        self.current_step = 0
        
        # Feature normalization
        self._normalize_features()
        
        # Initialize Redis cache if enabled
        self.use_cache = use_cache
        if use_cache:
            try:
                self.cache = RedisTaskCache(host=redis_host, port=redis_port)
            except Exception as e:
                print(f"Failed to connect to Redis: {e}")
                print("Using in-memory cache instead")
                self.cache = RedisTaskCache(use_redis=False)
        
        # Define observation space
        # Features: [data_size, cpu_cycles, memory_required, deadline,
        #            device_cpu_capacity, edge_cpu_capacity,
        #            network_bandwidth, network_latency]
        self.obs_dim = 8
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.obs_dim,), dtype=np.float32
        )
        
        # Define action space (binary: local or offload)
        self.action_space = spaces.Discrete(2)
        
        # Metrics
        self.reset_metrics()
    
    def _normalize_features(self):
        """Normalize dataset features to [0, 1] range"""
        # Create a mapping from expected column names to actual column names
        column_mapping = {
            'data_size': 'DataSize_kB',
            'cpu_cycles': 'CPU_Cycles_GHz',
            'memory_required': 'Memory_MB',
            'deadline': 'Deadline_s',
            'iot_cpu_power': 'Computational_Frequency',  # Using appropriate column
            'efaas_core_power': 'Min_CPU_Speed',        # Using appropriate column
            'bandwidth': 'DataSize_kB',                # Will be adjusted for bandwidth estimation
            'channel_gain': 'Priority_Score'           # Using as proxy for channel gain
        }
        
        # Store normalization parameters
        self.norm_params = {}
        
        # Create normalized columns
        for expected_col, actual_col in column_mapping.items():
            if actual_col in self.dataset.columns:
                # Get min and max values
                min_val = self.dataset[actual_col].min()
                max_val = self.dataset[actual_col].max()
                
                # Store normalization parameters
                self.norm_params[expected_col] = {
                    'min': min_val,
                    'max': max_val
                }
                
                # Normalize column
                if max_val > min_val:
                    self.dataset[expected_col + '_norm'] = (self.dataset[actual_col] - min_val) / (max_val - min_val)
                else:
                    self.dataset[expected_col + '_norm'] = 0.5  # Default value if all values are the same
            else:
                # If column doesn't exist, create a default normalized column
                print(f"Warning: Column '{actual_col}' not found in dataset. Using default values for '{expected_col}'.")
                self.dataset[expected_col + '_norm'] = 0.5
    
    def reset_metrics(self):
        """Reset environment metrics"""
        self.metrics = {
            'total_latency': 0.0,
            'energy_consumption': 0.0,
            'deadline_violations': 0,
            'offload_ratio': 0.0,
            'cache_hit_ratio': 0.0,
            'total_tasks': 0,
            'offloaded_tasks': 0,
            'cache_hits': 0
        }
    
    def reset(self):
        """
        Reset the environment for a new episode
        
        Returns:
            observation: Initial state
        """
        # Reset step counter
        self.current_step = 0
        
        # Reset metrics
        self.reset_metrics()
        
        # Reset cache
        if self.use_cache:
            self.cache.flush()
        
        # Get initial batch of tasks
        return self._get_next_observation()
    
    def _get_next_observation(self):
        """
        Get the next batch of tasks as observation
        
        Returns:
            observation: Batch of normalized task features
        """
        # Sample a batch of random tasks from the dataset
        batch_indices = np.random.choice(
            len(self.dataset), size=self.batch_size, replace=True
        )
        batch = self.dataset.iloc[batch_indices]
        self.current_batch = batch  # Store current batch for reference
        
        # Extract normalized features
        observations = []
        for _, task in batch.iterrows():
            # Create observation vector using the normalized columns we created
            obs = np.array([
                task.get('data_size_norm', 0.5),
                task.get('cpu_cycles_norm', 0.5),
                task.get('memory_required_norm', 0.5),
                task.get('deadline_norm', 0.5),
                task.get('iot_cpu_power_norm', 0.5),
                task.get('efaas_core_power_norm', 0.5),
                task.get('bandwidth_norm', 0.5),
                task.get('channel_gain_norm', 0.5)
            ], dtype=np.float32)
            
            observations.append(obs)
        
        return observations
    
    def _compute_execution_metrics(self, tasks, actions):
        """
        Compute execution metrics for the current batch
        
        Args:
            tasks: Batch of tasks
            actions: Batch of actions (0=local, 1=offload)
            
        Returns:
            rewards: Batch of rewards
            latencies: Batch of execution latencies
            deadline_violations: Batch of deadline violation flags
            cache_hits: Batch of cache hit flags
        """
        rewards = []
        latencies = []
        deadline_violations = []
        cache_hits = []
        
        # Track per-batch metrics to avoid accumulation errors
        batch_offloaded = 0
        batch_cache_hits = 0
        
        # Handle multi-agent scenario: If actions list is shorter than tasks,
        # we need to repeat actions to match the number of tasks
        if len(actions) < len(tasks):
            # Instead of extending with default actions (0), cycle through the provided actions
            num_actions = len(actions)
            actions = [actions[i % num_actions] for i in range(len(tasks))]
        
        for i, (_, task) in enumerate(tasks.iterrows()):
            action = actions[i]
            
            # Initialize variables that need to be defined in all code paths
            deadline = task['Deadline_s']
            latency = 0.0
            energy = 0.0
            deadline_violated = False
            
            # Check cache for similar task if offloading
            cache_hit = False
            if self.use_cache and action == 1:
                # Count as an offloaded task
                batch_offloaded += 1
                self.metrics['offloaded_tasks'] += 1
                
                # Create a key from task parameters - more specific for better training
                task_key = f"{task['DataSize_kB']}_{task['CPU_Cycles_GHz']}_{task['Memory_MB']}_{task['Deadline_s']}_{self.current_step % 100}"
                
                # Try to get result from cache
                cached_result = self.cache.get(task_key)
                if cached_result is not None:
                    latency, deadline_violated = cached_result
                    cache_hit = True
                    batch_cache_hits += 1
                    self.metrics['cache_hits'] += 1
                    # Set default energy value for cached results
                    energy = 0.1  # Small energy cost for cache lookup
            
            # Calculate execution metrics if no cache hit
            if not cache_hit:
                # Use actual dataset columns
                data_size = task['DataSize_kB']
                cpu_cycles = task['CPU_Cycles_GHz'] * 1e9  # Convert GHz to Hz
                
                # Local execution (on IoT device)
                if action == 0:
                    # Local execution time
                    iot_cpu_power = task.get('Computational_Frequency', 1.0)  # Default to 1.0 if not found
                    execution_time = cpu_cycles / (iot_cpu_power * 1e6)  # Adjust for units
                    
                    # Energy consumption for local execution
                    energy = 0.5 * iot_cpu_power * execution_time
                    
                    # Total latency is just execution time
                    latency = execution_time
                
                # Offload to edge
                else:
                    # Transmission time - using DataSize_kB for bandwidth calculation
                    bandwidth = 10 + (task['DataSize_kB'] % 90)  # Just as an example, generate a bandwidth value
                    transmission_time = (data_size * 8) / (bandwidth * 1e3)  # kB to kb and adjust for bandwidth units
                    
                    # Edge execution time
                    efaas_core_power = task.get('Min_CPU_Speed', 1.0)  # Default to 1.0 if not found
                    execution_time = cpu_cycles / (efaas_core_power * 1e9)  # Adjusted units
                    
                    # Total latency includes transmission and execution
                    latency = transmission_time + execution_time
                    
                    # Energy consumption for offloading
                    transmission_power = 0.5  # Watts
                    energy = transmission_power * transmission_time
                    
                    # Cache the result for future similar tasks
                    if self.use_cache:
                        deadline_violated = latency > deadline
                        self.cache.set(task_key, (latency, deadline_violated))
            
            # Check if deadline is violated
            deadline_violated = latency > deadline
            if deadline_violated:
                self.metrics['deadline_violations'] += 1
            
            # Calculate reward
            # Negative reward proportional to latency if deadline is met,
            # large penalty if deadline is violated
            if not deadline_violated:
                reward = -latency
            else:
                reward = -10.0 - latency
            
            # Add results to batch
            rewards.append(reward)
            latencies.append(latency)
            deadline_violations.append(float(deadline_violated))
            cache_hits.append(float(cache_hit))
            
            # Update environment metrics
            self.metrics['total_latency'] += latency
            self.metrics['energy_consumption'] += energy
        
        # Update total tasks counter
        self.metrics['total_tasks'] += len(tasks)
        
        return rewards, latencies, deadline_violations, cache_hits
    
    def step(self, actions):
        """
        Execute one step in the environment
        
        Args:
            actions: List of actions for the batch (0=local, 1=offload)
            
        Returns:
            next_observations: Batch of next states
            rewards: Batch of rewards
            dones: Batch of done flags
            info: Additional information
        """
        # Sample batch of tasks
        batch_indices = np.random.choice(
            len(self.dataset), size=self.batch_size, replace=True
        )
        batch = self.dataset.iloc[batch_indices]
        
        # Compute rewards and metrics
        rewards, latencies, deadline_violations, cache_hits = self._compute_execution_metrics(
            batch, actions
        )
        
        # Update current step
        self.current_step += 1
        
        # Check if episode is done
        dones = [self.current_step >= self.simulation_steps] * self.batch_size

        # Get next batch of tasks
        next_observations = self._get_next_observation()
        
        # Update offload ratio and cache hit ratio metrics
        if self.metrics['total_tasks'] > 0:
            self.metrics['offload_ratio'] = self.metrics['offloaded_tasks'] / self.metrics['total_tasks']
            
            if self.metrics['offloaded_tasks'] > 0:
                self.metrics['cache_hit_ratio'] = self.metrics['cache_hits'] / self.metrics['offloaded_tasks']
        
        # Return SARS tuple and info
        # For info dict, we just use the first task from the current batch
        task = batch.iloc[0]

        execution_time = task.get('ExecutionTime_s', 0.0)
        transmission_time = task.get('TransmissionTime_s', 0.0)
        setup_time = task.get('SetupTime_s', 0.0)
        waiting_time = task.get('WaitingTime_s', 0.0)

        computed_latency = execution_time + transmission_time + setup_time + waiting_time

        info = {
            'CPU_Cycles': [task.get('CPU_Cycles_GHz', 0.0)],
            'Min_CPU_Speed': [task.get('Min_CPU_Speed', 0.0)],
            'ExecutionTime_s': [execution_time],
            'DataSize_kB': [task.get('DataSize_kB', 0.0)],
            'TotalExecutionTime_s': [computed_latency],
            'Deadline_s': [task.get('Deadline_s', 0.0)],
            'latencies': [computed_latency],
            'deadline_violations': [1 if computed_latency > task.get('Deadline_s', float('inf')) else 0]
        }
        # print(f"[ENV] Step: {self.current_step}, Done: {self.current_step >= self.simulation_steps}")

        return next_observations, rewards, dones, info
    
    def render(self, mode='human'):
        """
        Render the environment
        
        Args:
            mode: Rendering mode
        """
        if self.current_step % 100 == 0:
            avg_latency = self.metrics['total_latency'] / max(1, self.metrics['total_tasks'])
            
            print(f"Step: {self.current_step}/{self.simulation_steps}")
            print(f"Average Latency: {avg_latency:.4f} s")
            print(f"Deadline Violations: {self.metrics['deadline_violations']}")
            print(f"Offload Ratio: {self.metrics['offload_ratio']:.4f}")
            print(f"Cache Hit Ratio: {self.metrics['cache_hit_ratio']:.4f}")
            print("-" * 40)
    
    def get_metrics(self):
        """
        Get current environment metrics
        
        Returns:
            metrics: Dictionary of environment metrics
        """
        return self.metrics