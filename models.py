from pydantic import BaseModel
from typing import Optional, List

class CompressRequest(BaseModel):
    """Request model for compression API"""
    
    prompts: Optional[List[str]] = []
    rate: Optional[float] = None
    target_token: Optional[int] = None
    target_context_level_rate: Optional[float] = None
    context_level_rate: Optional[float] = None
    context_level_target_token: Optional[int] = None
    chunk_end_tokens: Optional[str] = None
    query: Optional[str] = None

def build_compression_params(request: CompressRequest) -> dict:
    """Build compression parameters from request
    
    Args:
        request: CompressRequest object containing compression parameters
        
    Returns:
        dict: Dictionary of valid compression parameters
    """
    params = {}
    
    # Validate and add rate parameter
    if request.rate is not None and 0 < request.rate <= 1:
        params["rate"] = request.rate
    
    # Validate and add target_token parameter
    if request.target_token is not None and request.target_token > 0:
        params["target_token"] = request.target_token
    
    # Validate and add target_context_level_rate parameter
    if request.target_context_level_rate is not None and request.target_context_level_rate > 0:
        params["target_context_level_rate"] = request.target_context_level_rate
    
    # Validate and add context_level_rate parameter
    if request.context_level_rate is not None and 0 < request.context_level_rate <= 1:
        params["context_level_rate"] = request.context_level_rate
    
    # Validate and add context_level_target_token parameter
    if request.context_level_target_token is not None and request.context_level_target_token > 0:
        params["context_level_target_token"] = request.context_level_target_token
    
    # Validate and add chunk_end_tokens parameter
    if request.chunk_end_tokens is not None and request.chunk_end_tokens != "":
        params["chunk_end_tokens"] = request.chunk_end_tokens
    
    return params 