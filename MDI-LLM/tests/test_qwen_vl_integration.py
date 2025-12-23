#!/usr/bin/env python3
"""
Quick test script to verify Qwen2-VL integration components work.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_vision_processor():
    """Test the VisionProcessor module."""
    print("=" * 60)
    print("Testing VisionProcessor...")
    print("=" * 60)
    
    from sub.vision_processor import VisionProcessor
    from PIL import Image
    import numpy as np
    
    processor = VisionProcessor()
    print(f"✓ VisionProcessor initialized")
    print(f"  - Min pixels: {processor.min_pixels}")
    print(f"  - Max pixels: {processor.max_pixels}")
    print(f"  - Patch size: {processor.patch_size}")
    
    try:
        # Create a synthetic test image instead of downloading
        print(f"\nCreating synthetic test image...")
        test_array = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        image = Image.fromarray(test_array, 'RGB')
        print(f"✓ Test image created: {image.size} ({image.mode})")
        
        # Test resizing
        resized = processor.smart_resize(image)
        print(f"✓ Image resized to: {resized.size}")
        
        # Calculate patches
        num_patches = processor.get_num_patches(resized.size[1], resized.size[0])
        print(f"✓ Number of patches: {num_patches}")
        
        # Test preprocessing
        tensor = processor.preprocess_image(
            resized,
            device=torch.device('cpu'),
            dtype=torch.float32,
        )
        print(f"✓ Tensor created: {tensor.shape}")
        print(f"  - Dtype: {tensor.dtype}")
        print(f"  - Device: {tensor.device}")
        print(f"  - Value range: [{tensor.min().item():.3f}, {tensor.max().item():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test that Qwen2-VL config is loaded correctly."""
    print("\n" + "=" * 60)
    print("Testing Qwen2-VL Config...")
    print("=" * 60)
    
    from sub.model import Config
    
    try:
        config = Config.from_name("Qwen2-VL-2B-Instruct")
        print(f"✓ Config loaded: {config.name}")
        print(f"  - n_layer: {config.n_layer}")
        print(f"  - n_head: {config.n_head}")
        print(f"  - n_embd: {config.n_embd}")
        print(f"  - vocab_size: {config.vocab_size}")
        print(f"  - is_multimodal: {config.is_multimodal}")
        
        if config.is_multimodal:
            print(f"  - vision_encoder_embed_dim: {config.vision_encoder_embed_dim}")
            print(f"  - vision_encoder_depth: {config.vision_encoder_depth}")
            print(f"  - vision_patch_size: {config.vision_patch_size}")
            print(f"  - use_mrope: {config.use_mrope}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vision_encoder():
    """Test VisionEncoder initialization."""
    print("\n" + "=" * 60)
    print("Testing VisionEncoder...")
    print("=" * 60)
    
    from sub.model import Config, VisionEncoder
    
    try:
        config = Config.from_name("Qwen2-VL-2B-Instruct")
        encoder = VisionEncoder(config)
        print(f"✓ VisionEncoder initialized")
        print(f"  - Patch size: {encoder.patch_size}")
        print(f"  - Embed dim: {encoder.embed_dim}")
        print(f"  - Depth: {encoder.depth}")
        print(f"  - Num heads: {encoder.num_heads}")
        
        # Test forward pass with dummy input
        dummy_image = torch.randn(1, 3, 224, 224)
        print(f"\n  Testing forward pass with dummy image {dummy_image.shape}...")
        output = encoder(dummy_image)
        print(f"✓ Forward pass successful")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Expected patches: {(224 // encoder.patch_size) ** 2}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_starter_node():
    """Test StarterNode with multimodal support."""
    print("\n" + "=" * 60)
    print("Testing StarterNode (Multimodal)...")
    print("=" * 60)
    
    from sub.model import Config
    from sub.submodels import StarterNode
    
    try:
        config = Config.from_name("Qwen2-VL-2B-Instruct")
        starter = StarterNode(config, n_transf_layers=5, verb=False)
        print(f"✓ StarterNode initialized")
        print(f"  - Is multimodal: {starter.is_multimodal}")
        print(f"  - Has vision encoder: {hasattr(starter, 'vision_encoder')}")
        print(f"  - Has vision projection: {hasattr(starter, 'vision_projection')}")
        
        # Count parameters
        total_params = sum(p.numel() for p in starter.parameters())
        print(f"  - Total parameters: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Qwen2-VL Integration - Component Tests")
    print("=" * 60 + "\n")
    
    results = {
        "VisionProcessor": test_vision_processor(),
        "Config": test_config(),
        "VisionEncoder": test_vision_encoder(),
        "StarterNode": test_starter_node(),
    }
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! ✓")
    else:
        print("Some tests failed. See details above.")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
