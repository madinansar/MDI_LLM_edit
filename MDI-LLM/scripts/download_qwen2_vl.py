#!/usr/bin/env python3
"""
Download and prepare Qwen2-VL-2B-Instruct for distributed inference.
This script downloads the model from HuggingFace and splits it for multi-node deployment.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor
)

from sub.model import Config
from sub.utils import split_and_store_with_finisher


def download_qwen2_vl(
    model_name: str,
    save_dir: Path,
    n_nodes: int,
    dtype: str = "bfloat16"
) -> bool:
    """
    Download Qwen2-VL model and prepare for distributed inference.
    
    Args:
        model_name: HuggingFace model identifier
        save_dir: Directory to save the model
        n_nodes: Number of nodes for distributed inference
        dtype: Model dtype (bfloat16, float32, float16)
    
    Returns:
        True if successful, False otherwise
    """
    
    # Create save directory
    model_path = save_dir / model_name.split('/')[-1]
    model_path.mkdir(parents=True, exist_ok=True)
    
    # Set torch dtype
    torch_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = torch_dtype_map.get(dtype, torch.bfloat16)
    
    print(f"\nStep 1: Downloading model from HuggingFace...")
    try:
        # Download model
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="cpu"
        )
        print("✓ Model downloaded successfully")
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        return False
    
    # Download tokenizer and processor
    print("\nStep 2: Downloading tokenizer and processor...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        
        # Save tokenizer and processor
        tokenizer.save_pretrained(model_path)
        processor.save_pretrained(model_path)
        print("✓ Tokenizer and processor saved")
    except Exception as e:
        print(f"✗ Error with tokenizer/processor: {e}")
        return False
    
    # Skip saving full model to disk (it's huge and slow)
    # We only need the state_dict for splitting
    print("\nStep 3: Extracting state dict (skipping full model save to save time)...")
    state_dict = model.state_dict()
    print(f"✓ State dict extracted ({len(state_dict)} parameters)")
    
    # Analyze the architecture
    print("\nStep 4: Analyzing model architecture...")
    vision_keys = [k for k in state_dict.keys() if 'visual' in k.lower() or 'vision' in k.lower()]
    language_keys = [k for k in state_dict.keys() if k not in vision_keys]
    
    print(f"  Vision encoder keys: {len(vision_keys)}")
    print(f"  Language model keys: {len(language_keys)}")
    
    # Sample keys
    if vision_keys:
        print(f"\n  Sample vision keys:")
        for k in vision_keys[:5]:
            print(f"    - {k}")
    
    print(f"\n  Sample language keys:")
    for k in language_keys[:10]:
        print(f"    - {k}")
    
    # Split the model for distributed inference
    if n_nodes >= 2:
        print(f"\nStep 5: Splitting model for {n_nodes} nodes...")
        try:
            split_vl_model_from_hf(
                state_dict=state_dict,
                n_nodes=n_nodes,
                model_path=model_path,
                dtype=dtype
            )
            print(f"✓ Model split into {n_nodes} chunks")
        except Exception as e:
            print(f"✗ Error splitting model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "="*60)
    print("✓ Model preparation complete!")
    print(f"Model saved to: {model_path}")
    if n_nodes >= 2:
        print(f"Chunks saved to: {model_path}/chunks/{n_nodes}nodes_finisher/")
    print("="*60)
    
    return True


def map_hf_to_litgpt_key(hf_key: str) -> str:
    """
    Map HuggingFace Qwen2-VL key names to LitGPT format.
    
    Args:
        hf_key: Original HuggingFace key
    
    Returns:
        Mapped LitGPT key
    """
    # Handle vision encoder keys (keep as is or strip model. prefix)
    if "visual" in hf_key.lower():
        return hf_key
    
    # Handle language model keys
    key = hf_key
    
    # Embedding layer - handle both formats
    if "embed_tokens.weight" in key:
        return "transformer.wte.weight"
    
    # Transformer layers - handle model.language_model.layers or model.layers
    if ".layers." in key:
        # Extract parts
        parts = key.split(".")
        
        # Find the layer index
        layer_idx = None
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = parts[i + 1]
                rest_parts = parts[i + 2:]
                break
        
        if layer_idx is None:
            return key  # Can't parse, return original
        
        rest = ".".join(rest_parts)
        rest = ".".join(rest_parts)
        
        # Map component names
        component_map = {
            "self_attn.q_proj.weight": "attn.q_proj.weight",
            "self_attn.q_proj.bias": "attn.q_proj.bias",
            "self_attn.k_proj.weight": "attn.k_proj.weight",
            "self_attn.k_proj.bias": "attn.k_proj.bias",
            "self_attn.v_proj.weight": "attn.v_proj.weight",
            "self_attn.v_proj.bias": "attn.v_proj.bias",
            "self_attn.o_proj.weight": "attn.c_proj.weight",
            "self_attn.o_proj.bias": "attn.c_proj.bias",
            "mlp.gate_proj.weight": "mlp.gate_proj.weight",
            "mlp.up_proj.weight": "mlp.up_proj.weight",
            "mlp.down_proj.weight": "mlp.c_proj.weight",
            "input_layernorm.weight": "ln_1.weight",
            "post_attention_layernorm.weight": "ln_2.weight",
        }
        
        if rest in component_map:
            return f"transformer.h.{layer_idx}.{component_map[rest]}"
        else:
            # Keep original structure for unmapped keys
            return f"transformer.h.{layer_idx}.{rest}"
    
    # Final layer norm - handle both formats
    if "norm.weight" in key and "layers" not in key and "input" not in key and "post" not in key:
        return "transformer.ln_f.weight"
    
    # Language model head
    if key == "lm_head.weight":
        return "lm_head.weight"
    
    # If not matched, return original
    return key


def split_vl_model_from_hf(
    state_dict: dict,
    n_nodes: int,
    model_path: Path,
    dtype: str = "bfloat16"
) -> None:
    """
    Split vision-language model for distributed inference.
    
    Args:
        state_dict: Model state dictionary from HuggingFace
        n_nodes: Number of nodes
        model_path: Path to save model chunks
        dtype: Model dtype
    """
    
    # Separate vision and language weights
    vision_weights = {}
    language_weights = {}
    
    for key, value in state_dict.items():
        # Check for vision encoder keys (in Qwen2-VL they start with "model.visual.")
        if "visual" in key.lower():
            # Keep original key structure for vision
            vision_weights[key] = value
        else:
            # Map to LitGPT format for language model
            mapped_key = map_hf_to_litgpt_key(key)
            language_weights[mapped_key] = value
    
    print(f"\n  Separated weights:")
    print(f"    Vision encoder: {len(vision_weights)} tensors")
    print(f"    Language model: {len(language_weights)} tensors")
    
    # Use existing split function for language model
    print(f"\n  Splitting language model across {n_nodes} nodes...")
    split_and_store_with_finisher(
        model_params=language_weights,
        n_nodes=n_nodes,
        ckpt_dir=model_path
    )
    
    # Add vision encoder to starter chunk
    chunks_dir = model_path / "chunks" / f"{n_nodes}nodes_finisher"
    starter_chunk_path = chunks_dir / "model_starter.pth"
    
    print(f"\n  Adding vision encoder to starter chunk...")
    if starter_chunk_path.exists():
        starter_chunk = torch.load(starter_chunk_path, map_location="cpu")
        # Add vision weights
        for key, value in vision_weights.items():
            starter_chunk[key] = value
        # Save updated starter chunk
        torch.save(starter_chunk, starter_chunk_path)
        print(f"    ✓ Vision encoder added to {starter_chunk_path}")
    else:
        print(f"    ✗ Starter chunk not found at {starter_chunk_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare Qwen2-VL for distributed inference"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="src/checkpoints",
        help="Directory to save the model"
    )
    parser.add_argument(
        "--n-nodes",
        type=int,
        default=2,
        help="Number of nodes for distributed inference"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Downloading Qwen2-VL Model")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Save directory: {args.save_dir}")
    print(f"Number of nodes: {args.n_nodes}")
    print(f"Dtype: {args.dtype}")
    print("="*60)
    
    save_dir = Path(args.save_dir)
    
    success = download_qwen2_vl(
        model_name=args.model_name,
        save_dir=save_dir,
        n_nodes=args.n_nodes,
        dtype=args.dtype
    )
    
    if not success:
        print("\n✗ Model preparation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
