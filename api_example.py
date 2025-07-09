#!/usr/bin/env python3
"""
Simple API wrapper for Pandora Transformer (C implementation)
This demonstrates how to create a web API around your C transformer.

Usage:
    pip install fastapi uvicorn
    python api_example.py
    
Then visit: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import json
import os

app = FastAPI(
    title="Pandora Transformer API",
    description="A simple API wrapper for the C-based transformer implementation",
    version="1.0.0"
)

class GenerationRequest(BaseModel):
    prompt: str = "Hello world"
    max_length: int = 100

class GenerationResponse(BaseModel):
    generated_text: str
    model_info: dict

@app.get("/")
async def root():
    return {
        "message": "Pandora Transformer API", 
        "status": "running",
        "endpoints": ["/generate", "/health", "/docs"]
    }

@app.get("/health")
async def health_check():
    """Check if the transformer model is working"""
    try:
        # Test if the compiled transformer exists
        if not os.path.exists("./lattice_demo"):
            raise HTTPException(status_code=500, detail="Transformer binary not found. Run: gcc transformer_lattice.c -lm -o lattice_demo")
        
        # Quick test run
        result = subprocess.run(['./lattice_demo'], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Transformer execution failed: {result.stderr}")
        
        return {"status": "healthy", "model": "pandora-transformer-v1"}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Transformer execution timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """
    Generate text using the Pandora Transformer
    
    Note: The current C implementation is a demo that processes fixed embeddings.
    In a real implementation, you would:
    1. Tokenize the input prompt
    2. Pass tokens through the transformer
    3. Decode output tokens back to text
    """
    try:
        # Run the transformer (currently just processes demo embeddings)
        result = subprocess.run(['./lattice_demo'], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Generation failed: {result.stderr}")
        
        # In a real implementation, you'd process the prompt here
        # For now, we return the transformer output as a demonstration
        model_output = result.stdout.strip()
        
        return GenerationResponse(
            generated_text=f"Input: {request.prompt}\n\nTransformer Demo Output:\n{model_output}",
            model_info={
                "model_name": "pandora-transformer",
                "architecture": "transformer-lattice", 
                "layers": 2,
                "heads": 2,
                "d_model": 8,
                "note": "This is a pedagogical implementation. Real text generation requires tokenization and training."
            }
        )
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Generation timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get information about the transformer model"""
    return {
        "name": "Pandora Transformer",
        "version": "1.0.0",
        "architecture": {
            "type": "Transformer",
            "layers": 2,
            "attention_heads": 2, 
            "model_dimension": 8,
            "feed_forward_dimension": 16,
            "sequence_length": 4
        },
        "implementation": "C (pedagogical)",
        "features": [
            "Multi-head attention",
            "Feed-forward networks",
            "Layer normalization", 
            "Positional encoding",
            "Residual connections"
        ],
        "limitations": [
            "No tokenization",
            "No text generation",
            "Fixed demo embeddings",
            "Not trained on real data"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    
    # Check if transformer is compiled
    if not os.path.exists("./lattice_demo"):
        print("❌ Transformer not compiled!")
        print("Run: gcc transformer_lattice.c -lm -o lattice_demo")
        exit(1)
    
    print("🚀 Starting Pandora Transformer API...")
    print("📚 API Documentation will be available at /docs")
    print("🌐 Deploy this to your own domain - NOT OpenAI's platform!")
    
    # Get port from environment (for cloud deployment) or default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    # host="0.0.0.0" allows external connections (required for cloud deployment)
    uvicorn.run(app, host="0.0.0.0", port=port)