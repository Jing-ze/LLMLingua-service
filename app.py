import os
import uvicorn
import logging
import yaml
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from service.compressor_pool import CompressorPool
from service.models import CompressRequest, build_compression_params
from service.config import load_config, setup_logging

# Load configuration
config = load_config()

# Setup logging
setup_logging(config)
logger = logging.getLogger("llmlingua-service")

# Global variables
compressor_pool = None
default_model_path = config["model"]["default_model_path"]
default_device_map = config["model"]["default_device_map"]

def check_model_path(model_path):
    if not model_path:
        return False
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), model_path))
    return os.path.exists(model_path)

def initialize_compressor_pool():
    """Initialize connection pool"""
    global compressor_pool
    
    logger.info("Initializing LLMLingua compressor connection pool")
    
    try:
        # Determine model path to use
        if check_model_path(default_model_path):
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), default_model_path))
            logger.info(f"Using local model path: {model_path}")
            model_to_use = model_path
        else:
            logger.info("Local model path does not exist, using remote model")
            model_to_use = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
        
        # Initialize connection pool
        compressor_pool = CompressorPool(
            pool_size=config["model"]["pool_size"],  # Connection pool size
            model_name=model_to_use,
            device_map=default_device_map,
            use_llmlingua2=config["model"]["use_llmlingua2"],
        )
        
        logger.info("LLMLingua compressor connection pool initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize LLMLingua compressor connection pool: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Execute on startup
    logger.info("Application starting...")
    initialize_compressor_pool()
    logger.info("Application startup completed")
    
    yield  # During application runtime
    
    # Execute on shutdown
    logger.info("Application shutting down...")
    # Cleanup connection pool resources
    if compressor_pool is not None:
        compressor_pool.cleanup()
    logger.info("Application shutdown completed")

app = FastAPI(
    title="LLMLingua Compression Service", 
    description="API service for compressing prompts using LLMLingua",
    lifespan=lifespan
)

@app.post("/compress")
async def compress_prompt(request: CompressRequest):
    try:

        global compressor_pool
        global default_model_path
        global default_device_map

        # Check if connection pool is initialized
        if compressor_pool is None:
            raise HTTPException(status_code=503, detail="Connection pool not initialized, please try again later")

        params = build_compression_params(request)

        compressor, instance_id = compressor_pool.get_compressor()
        try:
            compressed_result = compressor.compress_prompt(
                context=request.prompts,
                **params,
            )
            compressed_prompt = compressed_result["compressed_prompt_list"]
            origin_tokens = compressed_result["origin_tokens"]
            compressed_tokens = compressed_result["compressed_tokens"]
            rate = compressed_result["rate"]
            return {
                "prompts": compressed_prompt,
                "original_tokens": origin_tokens,
                "compressed_tokens": compressed_tokens,
                "rate": rate,
            }
        finally:
            compressor_pool.release_compressor(instance_id)
    except Exception as e:
        error_msg = f"compress error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/health")
async def health_check():
    global compressor_pool
    
    if compressor_pool is None:
        return {
            "status": "initializing",
            "pool_status": None
        }
    
    pool_status = compressor_pool.get_status()
    return {
        "status": "healthy",
        "pool_status": pool_status
    }

if __name__ == "__main__":
    host = os.getenv("HOST", config["server"]["host"])
    port = int(os.getenv("PORT", config["server"]["port"]))
    
    logger.info(f"Starting compression server: {host}:{port}")
    try:
        uvicorn.run(
            "app:app", 
            host=host, 
            port=port, 
            reload=False,
            timeout_keep_alive=config["server"]["timeout_keep_alive"],  # 15 minutes keep-alive
            timeout_graceful_shutdown=config["server"]["timeout_graceful_shutdown"],  # Graceful shutdown timeout
            access_log=config["server"]["access_log"],
            workers=config["server"]["workers"],
        )
    except Exception as e:
        logger.error(f"start server error: {e}", exc_info=True)