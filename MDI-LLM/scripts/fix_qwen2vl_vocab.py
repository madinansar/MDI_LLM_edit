#!/usr/bin/env python3
"""
Fix Qwen2-VL checkpoint vocab size by padding embeddings to 152064.
"""

import torch
from pathlib import Path

def pad_embedding_weights(checkpoint_path: Path, vocab_size_target: int = 152064):
    """Pad embedding weights to target vocab size."""
    
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # Check current vocab size
    if 'transformer.wte.weight' in state_dict:
        current_shape = state_dict['transformer.wte.weight'].shape
        print(f"Current wte shape: {current_shape}")
        
        if current_shape[0] < vocab_size_target:
            # Pad to target size
            embed_dim = current_shape[1]
            padding_size = vocab_size_target - current_shape[0]
            
            print(f"Padding with {padding_size} zero embeddings...")
            padding = torch.zeros(padding_size, embed_dim, dtype=state_dict['transformer.wte.weight'].dtype)
            state_dict['transformer.wte.weight'] = torch.cat([
                state_dict['transformer.wte.weight'],
                padding
            ], dim=0)
            
            print(f"New wte shape: {state_dict['transformer.wte.weight'].shape}")
    
    # Check lm_head if exists
    if 'lm_head.weight' in state_dict:
        current_shape = state_dict['lm_head.weight'].shape
        print(f"Current lm_head shape: {current_shape}")
        
        if current_shape[0] < vocab_size_target:
            embed_dim = current_shape[1]
            padding_size = vocab_size_target - current_shape[0]
            
            print(f"Padding lm_head with {padding_size} zero rows...")
            padding = torch.zeros(padding_size, embed_dim, dtype=state_dict['lm_head.weight'].dtype)
            state_dict['lm_head.weight'] = torch.cat([
                state_dict['lm_head.weight'],
                padding
            ], dim=0)
            
            print(f"New lm_head shape: {state_dict['lm_head.weight'].shape}")
    
    # Save back
    print(f"Saving fixed checkpoint...")
    torch.save(state_dict, checkpoint_path)
    print("✓ Done")

if __name__ == "__main__":
    # Fix starter chunk
    starter_path = Path("src/checkpoints/Qwen2-VL-2B-Instruct/chunks/2nodes_finisher/model_starter.pth")
    print("="*60)
    print("Fixing starter chunk")
    print("="*60)
    pad_embedding_weights(starter_path)
    
    # Fix secondary chunk (has lm_head)
    secondary_path = Path("src/checkpoints/Qwen2-VL-2B-Instruct/chunks/2nodes_finisher/model_secondary0.pth")
    print("\n" + "="*60)
    print("Fixing secondary chunk")
    print("="*60)
    pad_embedding_weights(secondary_path)
