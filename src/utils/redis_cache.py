import subprocess
import pickle
import time
import os
import signal
import platform
import threading
import socket
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

class SimpleInMemoryCache:
    """Simple in-memory LRU cache as fallback when Redis is not available"""
    
    def __init__(self, maxsize=1000):
        self.cache = OrderedDict()
        self.maxsize = maxsize
    
    def get(self, key):
        """Get value from cache"""
        if key not in self.cache:
            return None
        
        # Move to end (most recently used)
        value = self.cache.pop(key)
        self.cache[key] = value
        
        return value
    
    def set(self, key, value):
        """Set value in cache"""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.pop(key)
            
        # Check if cache is full
        if len(self.cache) >= self.maxsize:
            # Remove oldest item
            self.cache.popitem(last=False)
            
        # Add new item
        self.cache[key] = value
    
    def flush(self):
        """Clear cache"""
        self.cache.clear()


class RedisTaskCache:
    """Redis-based cache for offloading task results"""
    
    def __init__(self, host='localhost', port=6379, ttl=3600, use_redis=True, maxsize=10000):
        """
        Initialize Redis cache
        
        Args:
            host (str): Redis host
            port (int): Redis port
            ttl (int): Time-to-live for cache entries in seconds
            use_redis (bool): Whether to use Redis (falls back to memory cache if False)
            maxsize (int): Maximum size of in-memory cache if Redis is not used
        """
        self.ttl = ttl
        self.use_redis = use_redis
        self.redis_client = None
        
        if use_redis:
            try:
                import redis
                self.redis_client = redis.Redis(host=host, port=port, decode_responses=False)
                # Test connection
                self.redis_client.ping()
                logger.info(f"Connected to Redis cache at {host}:{port}")
            except (ImportError, redis.ConnectionError) as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                logger.warning("Falling back to in-memory cache")
                self.use_redis = False
                
        if not self.use_redis:
            # Use in-memory cache as fallback
            self.memory_cache = SimpleInMemoryCache(maxsize=maxsize)
            logger.info("Using in-memory cache")
    
    def get(self, key):
        """
        Get value from cache
        
        Args:
            key (str): Cache key
            
        Returns:
            value: Cached value or None if not found
        """
        # Use Redis if available
        if self.use_redis and self.redis_client is not None:
            try:
                value = self.redis_client.get(key)
                if value is not None:
                    return pickle.loads(value)
            except Exception as e:
                logger.error(f"Error getting value from Redis: {e}")
                return None
        # Fall back to in-memory cache
        else:
            return self.memory_cache.get(key)
    
    def set(self, key, value, ttl=None):
        """
        Set value in cache
        
        Args:
            key (str): Cache key
            value: Value to cache
            ttl (int): Time-to-live in seconds (overrides instance ttl)
        """
        if ttl is None:
            ttl = self.ttl
            
        # Use Redis if available
        if self.use_redis and self.redis_client is not None:
            try:
                serialized_value = pickle.dumps(value)
                self.redis_client.setex(key, ttl, serialized_value)
            except Exception as e:
                logger.error(f"Error setting value in Redis: {e}")
        # Fall back to in-memory cache
        else:
            self.memory_cache.set(key, value)
    
    def flush(self):
        """Clear cache"""
        if self.use_redis and self.redis_client is not None:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                logger.error(f"Error flushing Redis cache: {e}")
        else:
            self.memory_cache.flush()


def start_redis_server():
    """Start Redis server in the background"""
    # Check if Redis is already running
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('localhost', 6379))
        s.close()
        print("Redis server is already running")
        return
    except (socket.error, ConnectionRefusedError):
        pass
    
    # Determine Redis executable based on OS
    redis_exec = 'redis-server'
    
    # Check if Redis is installed
    try:
        subprocess.run([redis_exec, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Redis server not found. Please install Redis.")
        if platform.system() == 'Windows':
            print("For Windows, download Redis from https://github.com/microsoftarchive/redis/releases")
        else:
            print("For Linux: sudo apt-get install redis-server")
            print("For Mac: brew install redis")
        return
    
    # Start Redis server
    print("Starting Redis server...")
    
    # Create config if needed
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config')
    os.makedirs(config_dir, exist_ok=True)
    
    config_file = os.path.join(config_dir, 'redis.conf')
    
    # Create default config if not exists
    if not os.path.exists(config_file):
        with open(config_file, 'w') as f:
            f.write("port 6379\n")
            f.write("bind 127.0.0.1\n")
            f.write("maxmemory 100mb\n")
            f.write("maxmemory-policy allkeys-lru\n")
            
    # Start Redis server as a subprocess
    if platform.system() == 'Windows':
        # On Windows, start Redis in background
        proc = subprocess.Popen(
            [redis_exec, config_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        # On Unix, use daemon mode
        proc = subprocess.Popen(
            [redis_exec, config_file, '--daemonize', 'yes'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    
    # Wait a moment to allow Redis to start
    time.sleep(1)
    
    # Check if Redis is running
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('localhost', 6379))
        s.close()
        print("Redis server started successfully")
    except (socket.error, ConnectionRefusedError):
        stdout, stderr = proc.communicate(timeout=2)
        print(f"Failed to start Redis server: {stderr.decode('utf-8', errors='ignore')}")
        return
    
    # Keep process running
    try:
        while True:
            print("Redis server is running. Press Ctrl+C to stop.")
            time.sleep(10)
    except KeyboardInterrupt:
        if platform.system() == 'Windows':
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
        print("Redis server stopped")