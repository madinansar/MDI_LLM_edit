#!/usr/bin/env python3

import os
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path

from sub.utils import load_from_hf, load_from_hf_direct, load_from_pt, split_and_store_with_finisher
from sub.utils.convert_hf_checkpoint import convert_hf_checkpoint
from sub.utils.download import download_from_hub
from sub.model import Config

docstring = """
Use this script to:
- Download weights, config and tokenizer info from Huggingface Hub (if not already downloaded)
- Store them in a local folder
- Partition them among a number of nodes, if specified
- Store the partitions at a specific location

Given the model name (required) and the checkpoint folder (optional - default:
`./checkpoints`), the model will be stored at:

    ./<checkpoint folder>/<hf model name>/

and the chunks will be stored in:

    ./<checkpoint folder>/<hf model name>/chunks/<N>nodes/

where `N` is the number of nodes for the partition contained in that subfolder.

Use --skip-convert to skip the HF to LitGPT conversion and keep the model in HF format.

---
"""

script_dir = os.path.dirname(__file__)


def is_vision_model(model_name: str) -> bool:
    """Check if model is a vision-language model."""
    vision_keywords = ['VL', 'vision', 'Vision', 'visual', 'Visual', 'multimodal', 'Multimodal']
    return any(keyword in model_name for keyword in vision_keywords)


def split_vl_model(state_dict: dict, n_nodes: int, model_path: Path, verb: bool = True) -> Path:
    """
    Split vision-language model with special handling for vision encoder.
    
    Strategy:
    - Vision encoder weights stay with starter node
    - Language model weights split normally across nodes
    
    Args:
        state_dict: Complete model state dict
        n_nodes: Number of nodes to split across
        model_path: Path to model checkpoint
        verb: Verbose output
    
    Returns:
        Path to chunks subfolder
    """
    import torch
    
    # Separate vision encoder weights from language model weights
    vision_keys = [
        'visual', 'vision_encoder', 'vision_tower', 'vision_model',
        'image_encoder', 'vision_projection', 'visual_projection'
    ]
    
    vision_weights = {}
    lm_weights = {}
    
    for key, value in state_dict.items():
        # Check if this key belongs to vision encoder
        is_vision = any(vk in key.lower() for vk in vision_keys)
        if is_vision:
            vision_weights[key] = value
        else:
            lm_weights[key] = value
    
    if verb:
        print(f"\n{'='*60}")
        print("Vision-Language Model Detected")
        print(f"{'='*60}")
        print(f"Vision encoder parameters: {len(vision_weights)}")
        print(f"Language model parameters: {len(lm_weights)}")
        
        # Calculate sizes
        vision_size = sum(p.numel() for p in vision_weights.values() if isinstance(p, torch.Tensor))
        lm_size = sum(p.numel() for p in lm_weights.values() if isinstance(p, torch.Tensor))
        total_size = vision_size + lm_size
        
        print(f"Vision encoder size: {vision_size:,} parameters ({vision_size/1e9:.2f}B)")
        print(f"Language model size: {lm_size:,} parameters ({lm_size/1e9:.2f}B)")
        print(f"Total size: {total_size:,} parameters ({total_size/1e9:.2f}B)")
        print(f"{'='*60}\n")
    
    # Split language model normally
    if verb:
        print("Splitting language model across nodes...")
    chunks_subfolder = split_and_store_with_finisher(lm_weights, n_nodes, model_path, verb=verb)
    
    # Add vision encoder to starter chunk
    starter_chunk_path = chunks_subfolder / "model_starter.pth"
    
    if verb:
        print(f"\nAdding vision encoder to starter node...")
        print(f"Loading starter chunk from: {starter_chunk_path}")
    
    starter_state = torch.load(starter_chunk_path, map_location='cpu')
    
    # Merge vision weights into starter state
    for key, value in vision_weights.items():
        starter_state[key] = value
    
    # Save updated starter chunk
    torch.save(starter_state, starter_chunk_path)
    
    if verb:
        print(f"✓ Vision encoder added to starter node")
        print(f"  Total parameters in starter: {len(starter_state):,}")
        print(f"  Saved to: {starter_chunk_path}")
    
    return chunks_subfolder


def main(args):
    os.makedirs(args.ckpt_folder, exist_ok=True)

    if Path(args.MODEL).is_dir():
        # Local model directory
        model_path = Path(args.MODEL)
        if args.skip_convert:
            # Load directly from HF format
            _, state_dict = load_from_hf_direct(model_path, args.device)
        else:
            if not (model_path / "lit_model.pth").exists() or not (
                model_path / "model_config.yaml"
            ).exists():
                # Need to convert the model to the Lit format
                convert_hf_checkpoint(checkpoint_dir=model_path, dtype=args.dtype)
            _, state_dict = load_from_pt(model_path, args.device)
    else:
        # Download from Huggingface
        model_path = Path(args.ckpt_folder) / args.MODEL
        
        if args.skip_convert:
            # Download without conversion
            download_from_hub(
                repo_id=args.MODEL,
                access_token=(
                    args.hf_token if args.hf_token is not None else os.getenv("HF_TOKEN")
                ),
                dtype=args.dtype,
                checkpoint_dir=args.ckpt_folder,
                model_name=args.model_name,
                convert_checkpoint=False,  # Don't convert
            )
            # Load directly from HF format
            _, state_dict = load_from_hf_direct(model_path, args.device)
        else:
            # Download and convert
            _, state_dict = load_from_hf(
                repo_id=args.MODEL,
                access_token=(
                    args.hf_token if args.hf_token is not None else os.getenv("HF_TOKEN")
                ),
                dtype=args.dtype,
                checkpoint_dir=args.ckpt_folder,
                model_name=args.model_name,
                device=args.device,
                convert_checkpoint=True,
            )

    print("Model was loaded!")

    # Split the model
    if not args.n_nodes:
        return

    assert state_dict is not None
    
    # Check if this is a vision-language model
    is_vl_model = is_vision_model(args.MODEL)
    
    if is_vl_model:
        print(f"\n🔍 Detected vision-language model: {args.MODEL}")
        chunks_subfolder = split_vl_model(state_dict, args.n_nodes, model_path, verb=True)
    else:
        chunks_subfolder = split_and_store_with_finisher(state_dict, args.n_nodes, model_path, verb=True)

    print(f"\n✓ Done! The chunks have been written to {chunks_subfolder}")


if __name__ == "__main__":
    parser = ArgumentParser(description=docstring, formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        "MODEL",
        type=str,
        help="""model to be downloaded - it should correspond to a local folder
        containing a model or to a Huggingface Hub model;""",
    )

    parser.add_argument(
        "--ckpt-folder",
        type=Path,
        default=Path(os.path.join(script_dir, "checkpoints")),
        help="""subfolder where the model directory will be placed; the model files
        will be found at `<ckpt_folder>/<hf_model_name>/`""",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="""allows to specify a different config name to use for this MODEL,
        allowing to download alternative weights for the same architecture""",
    )
    parser.add_argument(
        "--n-nodes",
        type=int,
        help="""number of nodes among which to partition the model - if not specified,
        the partition will not be performed""",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="""Huggingface Hub token to access restricted/private workspaces;
        not required if the HF_TOKEN env variable is set.""",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="data type of downloaded weights - they will be quantized if necessary",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="torch device where to load model and tensors (default: cpu)",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="skip HF to LitGPT conversion (keep original HF format)",
    )

    args = parser.parse_args()
    main(args)