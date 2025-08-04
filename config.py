import yaml
import logging

# Default configuration
DEFAULT_CONFIG = {
    "server": {
        "host": "0.0.0.0",
        "port": 8000,
        "timeout_keep_alive": 900,
        "timeout_graceful_shutdown": 900,
        "access_log": True,
        "workers": 4
    },
    "model": {
        "default_model_path": "./models/llmlingua-2-xlm-roberta-large-meetingbank",
        "default_device_map": "cuda",
        "pool_size": 2,
        "use_llmlingua2": True
    },
    "logging": {
        "level": "INFO"
    }
}

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Configuration file {config_path} not found, using default values")
        return DEFAULT_CONFIG
    except Exception as e:
        print(f"Error loading configuration: {e}, using default values")
        return DEFAULT_CONFIG

def setup_logging(config: dict):
    """Setup logging configuration
    
    Args:
        config: Configuration dictionary
    """
    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"]),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    ) 