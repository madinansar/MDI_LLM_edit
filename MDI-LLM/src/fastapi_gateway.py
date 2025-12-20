#!/usr/bin/env python3
"""
FastAPI Gateway for MDI-LLM Distributed Inference

This gateway accepts HTTP requests with prompts and forwards them to the
distributed inference system (starter + secondary nodes).

Usage:
    1. Start the secondary node first:
       ./src/secondary.py --chunk ./src/checkpoints/meta-llama/Llama-3.2-1B/chunks/2nodes/model_secondary0.pth --nodes-config ./src/settings_distr/config_m3pro.json 0 -v
    
    2. Start this FastAPI gateway:
       uvicorn fastapi_gateway:app --host 0.0.0.0 --port 8000
    
    3. Send requests:
       curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '{"prompt": "What is the capital of Turkey?", "max_tokens": 100, "n_samples": 1}'
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import os
import sys
from pathlib import Path
from typing import Optional, List
import logging

# Add src directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from sub.model_dist import GPTDistributed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="MDI-LLM Gateway",
    description="Gateway for distributed LLM inference",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration (can be overridden via environment variables)
DEFAULT_CONFIG = {
    #"ckpt_dir": "./src/checkpoints/meta-llama/Llama-3.2-1B",
    "ckpt_dir": "./src/checkpoints/Qwen/Qwen3-1.7B",
    "nodes_config": "./src/settings_distr/config_m3pro.json",
    "device": "cpu",  # Auto-detect
    "dtype": "bfloat16",  # Qwen3's native dtype - bfloat16 works on CPU (just not with autocast)
    "sequence_length": 1024,
    "default_max_tokens": 100,
    "default_n_samples": 1,
}

class GenerateRequest(BaseModel):
    """Request model for text generation"""
    prompt: str = Field(..., description="Input prompt for text generation")
    max_tokens: int = Field(
        DEFAULT_CONFIG["default_max_tokens"], 
        ge=1, 
        le=2048, 
        description="Maximum number of tokens to generate"
    )
    n_samples: int = Field(
        DEFAULT_CONFIG["default_n_samples"], 
        ge=1, 
        le=10, 
        description="Number of samples to generate"
    )
    temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (not yet implemented in backend)"
    )
    top_k: Optional[int] = Field(
        None,
        ge=1,
        description="Top-k sampling (not yet implemented in backend)"
    )

class GenerateResponse(BaseModel):
    """Response model for text generation"""
    samples: List[str] = Field(..., description="Generated text samples")
    generation_time: Optional[float] = Field(None, description="Total generation time in seconds")
    tokens_generated: int = Field(..., description="Number of tokens generated per sample")
    model: str = Field(..., description="Model name used")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "MDI-LLM Gateway",
        "version": "1.0.0",
        "model": DEFAULT_CONFIG["ckpt_dir"]
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    ckpt_path = Path(DEFAULT_CONFIG["ckpt_dir"])
    config_path = Path(DEFAULT_CONFIG["nodes_config"])
    
    return {
        "status": "healthy",
        "model_exists": ckpt_path.exists(),
        "config_exists": config_path.exists(),
        "device": DEFAULT_CONFIG["device"] or "auto",
        "dtype": DEFAULT_CONFIG["dtype"],
        "max_sequence_length": DEFAULT_CONFIG["sequence_length"]
    }

@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    """
    Generate text based on the input prompt using distributed inference.
    
    The secondary node(s) must be running before calling this endpoint.
    """
    try:
        logger.info(f"Received generation request: prompt='{request.prompt[:50]}...', max_tokens={request.max_tokens}, n_samples={request.n_samples}")
        
        # Resolve paths to absolute paths
        ckpt_path = Path(DEFAULT_CONFIG["ckpt_dir"]).resolve()
        config_path = Path(DEFAULT_CONFIG["nodes_config"]).resolve()
        
        if not ckpt_path.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Checkpoint directory not found: {ckpt_path}"
            )
        if not config_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Nodes config file not found: {config_path}"
            )
        
        # Initialize distributed model (creates a new starter instance)
        logger.info("Initializing GPTDistributed...")
        gpt_distr = GPTDistributed(
            node_type="starter",
            config_file=config_path,
            ckpt_dir=ckpt_path,
            device=DEFAULT_CONFIG["device"],
            dtype=DEFAULT_CONFIG["dtype"],
            model_seq_length=DEFAULT_CONFIG["sequence_length"],
            verb=False,  # Disable verbose debug output in production
            plots=False,  # Disable plots in API mode
        )
        
        # Run generation
        logger.info("Starting generation...")
        
        # Configure nodes first
        logger.info("Configuring secondary nodes...")
        if not gpt_distr.configure_nodes(n_samples=request.n_samples):
            raise HTTPException(status_code=500, detail="Failed to initialize network nodes")
        logger.info("Secondary nodes configured successfully")
        
        # Launch generation directly to get both text and timing
        logger.info("Launching generation...")
        out_text, time_gen = gpt_distr.gpt_serv.launch_starter(
            request.n_samples,
            request.max_tokens,
            request.prompt
        )
        
        # Extract generation time
        generation_time = None
        if time_gen and len(time_gen) > 0:
            generation_time = time_gen[-1][1]
        
        logger.info(f"Generation completed in {generation_time:.2f}s" if generation_time else "Generation completed")
        
        # Stop nodes
        logger.info("Stopping nodes...")
        gpt_distr.stop_nodes()
        
        return GenerateResponse(
            samples=out_text,
            generation_time=generation_time,
            tokens_generated=request.max_tokens,
            model=gpt_distr.full_model_name
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/configure")
async def configure(
    ckpt_dir: Optional[str] = None,
    nodes_config: Optional[str] = None,
    sequence_length: Optional[int] = None,
):
    """
    Update gateway configuration.
    Note: This updates the default config but doesn't reload existing models.
    """
    if ckpt_dir:
        DEFAULT_CONFIG["ckpt_dir"] = ckpt_dir
    if nodes_config:
        DEFAULT_CONFIG["nodes_config"] = nodes_config
    if sequence_length:
        DEFAULT_CONFIG["sequence_length"] = sequence_length
    
    return {
        "status": "updated",
        "config": DEFAULT_CONFIG
    }

if __name__ == "__main__":
    import uvicorn
    
    # Enable MPS fallback for unsupported operations
    import os
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # Parse command line arguments for configuration
    import argparse
    parser = argparse.ArgumentParser(description="FastAPI Gateway for MDI-LLM")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--ckpt", type=str, help="Path to model checkpoint directory")
    parser.add_argument("--nodes-config", type=str, help="Path to nodes configuration file")
    parser.add_argument("--sequence-length", type=int, help="Maximum sequence length")
    args = parser.parse_args()
    
    # Update config from CLI args
    if args.ckpt:
        DEFAULT_CONFIG["ckpt_dir"] = args.ckpt
    if args.nodes_config:
        DEFAULT_CONFIG["nodes_config"] = args.nodes_config
    if args.sequence_length:
        DEFAULT_CONFIG["sequence_length"] = args.sequence_length
    
    logger.info(f"Starting FastAPI Gateway on {args.host}:{args.port}")
    logger.info(f"Model: {DEFAULT_CONFIG['ckpt_dir']}")
    logger.info(f"Config: {DEFAULT_CONFIG['nodes_config']}")
    
    uvicorn.run(app, host=args.host, port=args.port)

 