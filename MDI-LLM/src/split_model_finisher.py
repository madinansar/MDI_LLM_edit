#!/usr/bin/env python3
"""
Split an existing downloaded model into 3 chunks with finisher pattern.
The last node will contain the final layers, layer norm, and output head.
"""

import os
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sub.utils import load_from_pt, split_and_store_with_finisher

# Configuration
# MODEL_PATH = Path("./src/checkpoints/meta-llama/Llama-3.2-1B")
MODEL_PATH = Path("./src/checkpoints/Qwen/Qwen3-1.7B")

N_NODES = 3
DEVICE = "cpu"  # Use CPU for splitting to save memory

def main():
    print("="*60)
    print("Splitting Existing Model into 3 Nodes (Finisher Pattern)")
    print("="*60)
    print()
    
    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"❌ Model not found at: {MODEL_PATH}")
        print("   Please download the model first.")
        return 1
    
    lit_model = MODEL_PATH / "lit_model.pth"
    if not lit_model.exists():
        print(f"❌ lit_model.pth not found at: {MODEL_PATH}")
        print("   The model may not be properly converted.")
        return 1
    
    print(f"✅ Found model at: {MODEL_PATH}")
    print(f"📦 Loading model from disk...")
    print()
    
    # Load the existing model
    config, state_dict = load_from_pt(str(MODEL_PATH), DEVICE)
    
    print(f"✅ Model loaded successfully!")
    print(f"   Model: {config.name}")
    print(f"   Layers: {config.n_layer}")
    print()
    
    # Split into 3 nodes
    print(f"🔪 Splitting into {N_NODES} nodes (with finisher)...")
    print(f"   Starter: layers + embeddings only")
    print(f"   Secondary 0: middle layers")
    print(f"   Secondary 1 (Finisher): final layers + LM head + output")
    print()
    
    chunks_subfolder = split_and_store_with_finisher(state_dict, N_NODES, MODEL_PATH, verb=True)
    
    print()
    print("="*60)
    print("✅ Split Complete!")
    print("="*60)
    print()
    print(f"Chunks saved to: {chunks_subfolder}")
    print()
    print("Files created:")
    print(f"  - {chunks_subfolder}/model_starter.pth")
    print(f"  - {chunks_subfolder}/model_secondary0.pth")
    print(f"  - {chunks_subfolder}/model_secondary1.pth (finisher)")
    print()
    print("Next steps:")
    print("  1. Terminal 1: ./start_secondary0_3nsodes.sh")
    print("  2. Terminal 2: ./start_secondary1_3nodes.sh")
    print("  3. Terminal 3: python src/fastapi_gateway.py --nodes-config ./src/settings_distr/config_m3pro_3nodes.json --sequence-length 1024")
    print()
    
    return 0

if __name__ == "__main__":
    exit(main())

 