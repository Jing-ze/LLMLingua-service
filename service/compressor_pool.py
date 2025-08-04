import threading
import time
import logging
from lingua import PromptCompressor

logger = logging.getLogger("llmlingua-service")

class CompressorPool:
    """Connection pool for managing multiple compressor instances"""
    
    def __init__(self, pool_size=4, model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank", 
                 device_map="cpu", use_llmlingua2=True):
        """
        Initialize compressor pool
        
        Args:
            pool_size: Number of compressor instances in the pool
            model_name: Name or path of the model to load
            device_map: Device to load the model on (cpu, cuda, etc.)
            use_llmlingua2: Whether to use LLMLingua2 compressor
        """
        self.pool_size = pool_size
        self.model_name = model_name
        self.device_map = device_map
        self.use_llmlingua2 = use_llmlingua2
        
        self.pool = []
        self.lock = threading.Lock()
        self.available = []
        self.in_use = set()
        
        logger.info(f"Initializing connection pool, pool size: {pool_size}")
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize connection pool"""
        try:
            for i in range(self.pool_size):
                compressor = PromptCompressor(
                    model_name=self.model_name,
                    device_map=self.device_map,
                    use_llmlingua2=self.use_llmlingua2,
                )
                self.pool.append(compressor)
                self.available.append(i)
                logger.info(f"Connection pool instance {i} initialized successfully")
            
            logger.info(f"Connection pool initialization completed, total {self.pool_size} instances")
        except Exception as e:
            logger.error(f"Connection pool initialization failed: {e}")
            raise
    
    def get_compressor(self, timeout=30):
        """
        Get an available compressor instance
        
        Args:
            timeout: Maximum time to wait for an available instance
            
        Returns:
            tuple: (compressor_instance, instance_id)
            
        Raises:
            Exception: When no compressor is available within timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.lock:
                if self.available:
                    instance_id = self.available.pop()
                    self.in_use.add(instance_id)
                    logger.debug(f"Acquired connection pool instance {instance_id}, remaining available: {len(self.available)}")
                    return self.pool[instance_id], instance_id
            
            # Wait for a while before retrying
            time.sleep(0.1)
        
        logger.warning(f"Connection pool exhausted, timeout after {timeout} seconds")
        raise Exception("Connection pool exhausted, please try again later")
    
    def release_compressor(self, instance_id):
        """
        Release compressor instance back to connection pool
        
        Args:
            instance_id: ID of the instance to release
        """
        with self.lock:
            if instance_id in self.in_use:
                self.in_use.remove(instance_id)
                self.available.append(instance_id)
                logger.debug(f"Released connection pool instance {instance_id}, currently available: {len(self.available)}")
    
    def update_model(self, model_name=None, device_map=None):
        """
        Update model configuration for all instances in connection pool
        
        Args:
            model_name: New model name or path
            device_map: New device mapping
        """
        if model_name is not None:
            self.model_name = model_name
        if device_map is not None:
            self.device_map = device_map
        
        logger.info(f"Updating connection pool model configuration: model_name={self.model_name}, device_map={self.device_map}")
        
        # Wait for all instances to be released
        while True:
            with self.lock:
                if not self.in_use:
                    break
            time.sleep(0.1)
        
        # Reinitialize connection pool
        self.pool.clear()
        self.available.clear()
        self.in_use.clear()
        self._initialize_pool()
    
    def get_status(self):
        """
        Get connection pool status
        
        Returns:
            dict: Status information including pool size, available instances, etc.
        """
        with self.lock:
            return {
                "pool_size": self.pool_size,
                "available": len(self.available),
                "in_use": len(self.in_use),
                "model_name": self.model_name,
                "device_map": self.device_map
            }
    
    def cleanup(self):
        """Clean up resources when shutting down"""
        logger.info("Cleaning up connection pool resources")
        with self.lock:
            self.pool.clear()
            self.available.clear()
            self.in_use.clear()
        logger.info("Connection pool cleanup completed") 