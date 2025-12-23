# Qwen2-VL-2B-Instruct Integration Guide

## Overview

This guide outlines the steps needed to integrate Qwen/Qwen2-VL-2B-Instruct (a vision-language model) into the MDI-LLM distributed inference system. The VL model differs from text-only models in several key ways:

1. **Vision Encoder**: Processes images into visual embeddings
2. **Multimodal Inputs**: Handles both text and images
3. **Dynamic Visual Tokens**: Images are converted to variable numbers of tokens based on resolution
4. **M-ROPE**: Uses Multimodal Rotary Position Embeddings instead of standard RoPE

## Architecture Changes Required

### 1. Model Components

**Qwen2-VL Architecture**:
- **Vision Encoder** (ViT-based): Processes images → visual tokens
- **Language Model Backbone**: Standard transformer layers (similar to Qwen2)
- **Multimodal Fusion**: Integrates visual tokens with text tokens
- **M-ROPE**: 3D positional encoding (1D text + 2D vision + 3D video)

**Distribution Strategy**:
```
Node Type         | Components
------------------|------------------------------------------
Starter           | Embedding layer + Vision Encoder + First N transformer layers
Secondary(ies)    | Middle transformer layers
Finisher          | Last N transformer layers + LM head
```

### 2. Key Modifications Needed

#### A. Image Preprocessing Pipeline
```python
# New module: src/sub/vision_processor.py
- Load images from file/URL/base64
- Resize to dynamic resolution (maintain aspect ratio)
- Convert to visual tokens using vision encoder
- Merge with text embeddings
```

#### B. Model Architecture Updates
```python
# src/sub/model.py
- Add VisionEncoder class
- Add Qwen2VLConfig (extends Config)
- Update positional embeddings to support M-ROPE
- Handle variable-length visual token sequences
```

#### C. Distributed Node Updates
```python
# src/sub/submodels.py
- StarterNode: Include vision encoder
- Handle multimodal input sequences
- Pass combined (text + vision) embeddings to secondary
```

#### D. API Gateway Updates
```python
# src/fastapi_gateway.py
- Accept image inputs (file upload, URL, base64)
- New endpoint schema with multimodal content
- Image preprocessing before inference
```

## Implementation Steps

### Phase 1: Install Dependencies

```bash
# Upgrade transformers to support Qwen2-VL
pip install --upgrade transformers
pip install qwen-vl-utils  # Official utilities
pip install pillow>=10.0.0
pip install torchvision
pip install decord  # For video support (optional)
```

### Phase 2: Create Vision Processing Module

Create `src/sub/vision_processor.py`:

```python
"""
Vision preprocessing for Qwen2-VL models.
Handles image loading, resizing, and conversion to visual tokens.
"""
import torch
from PIL import Image
from torchvision import transforms
import requests
from io import BytesIO
import base64
from typing import Union, List, Optional
from pathlib import Path

class VisionProcessor:
    def __init__(
        self,
        min_pixels: int = 256 * 28 * 28,  # 200K pixels
        max_pixels: int = 1280 * 28 * 28,  # 1M pixels
        patch_size: int = 28,
    ):
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        
    def load_image(self, image_input: Union[str, Path, bytes]) -> Image.Image:
        """Load image from file path, URL, or base64 string."""
        if isinstance(image_input, (str, Path)):
            image_str = str(image_input)
            if image_str.startswith('http://') or image_str.startswith('https://'):
                # URL
                response = requests.get(image_str)
                image = Image.open(BytesIO(response.content))
            elif image_str.startswith('data:image'):
                # Base64
                base64_data = image_str.split(',')[1]
                image_data = base64.b64decode(base64_data)
                image = Image.open(BytesIO(image_data))
            elif image_str.startswith('file://'):
                # File path with file:// prefix
                image = Image.open(image_str[7:])
            else:
                # Regular file path
                image = Image.open(image_str)
        else:
            # Bytes
            image = Image.open(BytesIO(image_input))
        
        return image.convert('RGB')
    
    def smart_resize(
        self,
        image: Image.Image,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ) -> Image.Image:
        """
        Resize image to fit within pixel range while maintaining aspect ratio.
        Result dimensions are multiples of patch_size.
        """
        min_pixels = min_pixels or self.min_pixels
        max_pixels = max_pixels or self.max_pixels
        
        width, height = image.size
        current_pixels = width * height
        
        # Calculate scaling factor
        if current_pixels > max_pixels:
            scale = (max_pixels / current_pixels) ** 0.5
        elif current_pixels < min_pixels:
            scale = (min_pixels / current_pixels) ** 0.5
        else:
            scale = 1.0
        
        # New dimensions (rounded to patch_size multiples)
        new_width = int(width * scale)
        new_height = int(height * scale)
        new_width = (new_width // self.patch_size) * self.patch_size
        new_height = (new_height // self.patch_size) * self.patch_size
        
        # Ensure at least one patch
        new_width = max(new_width, self.patch_size)
        new_height = max(new_height, self.patch_size)
        
        if (new_width, new_height) != (width, height):
            image = image.resize((new_width, new_height), Image.BICUBIC)
        
        return image
    
    def preprocess_image(
        self,
        image: Image.Image,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Convert PIL image to tensor with normalization."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        
        image_tensor = transform(image).to(device=device, dtype=dtype)
        return image_tensor.unsqueeze(0)  # Add batch dimension
```

### Phase 3: Add Vision Encoder to Model

Update `src/sub/model.py`:

```python
# Add to imports
from torchvision.models import vision_transformer

class VisionEncoder(nn.Module):
    """Vision encoder for Qwen2-VL based on Vision Transformer."""
    
    def __init__(
        self,
        embed_dim: int = 1536,  # Qwen2-VL-2B uses 1536 for vision
        depth: int = 24,
        num_heads: int = 16,
        patch_size: int = 28,
        image_size: int = 28 * 40,  # Variable, this is default
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Projection to language model dimension
        self.visual_projection = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [batch, channels, height, width]
        Returns:
            visual_embeds: [batch, num_patches, embed_dim]
        """
        # Patch embedding
        x = self.patch_embed(pixel_values)  # [B, embed_dim, H/P, W/P]
        B, C, H, W = x.shape
        
        # Flatten patches
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # Add 2D positional embeddings (part of M-ROPE)
        # Simplified version - full M-ROPE is more complex
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Project to language model dimension
        x = self.visual_projection(x)
        
        return x


class Qwen2VLConfig(Config):
    """Extended config for Qwen2-VL models."""
    
    vision_encoder_embed_dim: int = 1536
    vision_encoder_depth: int = 24
    vision_encoder_num_heads: int = 16
    vision_patch_size: int = 28
    vision_start_token_id: int = 151652  # <|vision_start|>
    vision_end_token_id: int = 151653    # <|vision_end|>
    image_token_id: int = 151655         # <|image_pad|>
    
    # M-ROPE configuration
    use_mrope: bool = True  # Use multimodal RoPE
    mrope_section_text: int = 1
    mrope_section_image: int = 2
    mrope_section_video: int = 3


class GPTMultimodal(GPT):
    """
    Extended GPT model with vision capabilities.
    Inherits from the base GPT and adds vision encoder.
    """
    
    def __init__(self, config: Qwen2VLConfig) -> None:
        super().__init__(config)
        
        # Add vision encoder
        self.vision_encoder = VisionEncoder(
            embed_dim=config.vision_encoder_embed_dim,
            depth=config.vision_encoder_depth,
            num_heads=config.vision_encoder_num_heads,
            patch_size=config.vision_patch_size,
        )
        
        # Vision-to-language projection
        self.vision_projection = nn.Linear(
            config.vision_encoder_embed_dim,
            config.n_embd
        )
        
    def forward(
        self,
        idx: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        vision_token_mask: Optional[torch.Tensor] = None,
        max_seq_length: Optional[int] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional image features.
        
        Args:
            idx: Text token IDs [batch, seq_len]
            image_features: Pre-computed image features [batch, num_patches, vision_dim]
            vision_token_mask: Boolean mask indicating vision token positions
            max_seq_length: Maximum sequence length
            input_pos: Position IDs
        """
        # Get text embeddings
        x = self.transformer.wte(idx)  # [batch, seq_len, n_embd]
        
        # Merge vision features if provided
        if image_features is not None and vision_token_mask is not None:
            # Project vision features to language model dimension
            vision_embeds = self.vision_projection(image_features)
            
            # Replace vision token positions with actual vision embeddings
            x[vision_token_mask] = vision_embeds.flatten(0, 1)
        
        # Continue with standard transformer forward
        if input_pos is not None:
            rope = self.build_rope_cache(idx.device)
            rope = rope[input_pos]
        else:
            rope = self.rope_cache
        
        if max_seq_length:
            mask = self.mask_cache[:max_seq_length, :max_seq_length]
        else:
            mask = self.mask_cache
        
        # Transformer blocks
        for block in self.transformer.h:
            x = block(x, rope, mask, max_seq_length, input_pos)
        
        x = self.transformer.ln_f(x)
        return self.lm_head(x)
```

### Phase 4: Update Distributed Nodes

Update `src/sub/submodels.py`:

```python
class StarterNodeMultimodal(StarterNode):
    """
    Starter node with vision encoder support.
    Handles multimodal inputs and sends combined embeddings downstream.
    """
    
    def __init__(self, config: Qwen2VLConfig, n_transformer_layers: int, **kwargs):
        super().__init__(config, n_transformer_layers, **kwargs)
        
        # Add vision encoder
        self.vision_encoder = VisionEncoder(
            embed_dim=config.vision_encoder_embed_dim,
            depth=config.vision_encoder_depth,
            num_heads=config.vision_encoder_num_heads,
            patch_size=config.vision_patch_size,
        )
        
        # Vision projection
        self.vision_projection = nn.Linear(
            config.vision_encoder_embed_dim,
            config.n_embd
        )
    
    def forward(
        self,
        idx: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        vision_token_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with optional vision input.
        
        Args:
            idx: Token IDs including vision placeholders
            pixel_values: Image tensors [batch, channels, height, width]
            vision_token_mask: Mask indicating where to insert vision features
        """
        # Get text embeddings
        x = self.transformer.wte(idx)
        
        # Process vision if provided
        if pixel_values is not None and vision_token_mask is not None:
            # Encode images
            vision_features = self.vision_encoder(pixel_values)
            # Project to language dimension
            vision_embeds = self.vision_projection(vision_features)
            # Insert vision embeddings at placeholder positions
            x[vision_token_mask] = vision_embeds.flatten(0, 1)
        
        # Apply position embeddings (M-ROPE)
        rope = self._build_mrope_cache(idx, vision_token_mask)
        
        # Pass through transformer layers
        for block in self.transformer.h.values():
            x = block(x, rope, self.mask_cache)
        
        return x  # Send to next node
```

### Phase 5: Update API Gateway

Update `src/fastapi_gateway.py`:

```python
from typing import List, Optional, Union
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, HTTPException
import base64

class MultimodalContent(BaseModel):
    """Single content item (text or image)"""
    type: str  # "text" or "image"
    text: Optional[str] = None  # For text type
    image: Optional[str] = None  # For image type (URL, base64, or file path)
    
class MultimodalMessage(BaseModel):
    """Message with multimodal content"""
    role: str = "user"
    content: List[MultimodalContent]

class GenerateRequestMultimodal(BaseModel):
    """Request with multimodal support"""
    messages: List[MultimodalMessage]
    max_tokens: int = 100
    n_samples: int = 1
    temperature: Optional[float] = 0.7

@app.post("/generate/multimodal")
async def generate_multimodal(
    request: GenerateRequestMultimodal,
    files: Optional[List[UploadFile]] = File(None)
):
    """
    Generate text from multimodal input (text + images).
    
    Example request:
    {
      "messages": [
        {
          "role": "user",
          "content": [
            {"type": "image", "image": "https://example.com/image.jpg"},
            {"type": "text", "text": "What's in this image?"}
          ]
        }
      ],
      "max_tokens": 100
    }
    """
    try:
        # Parse multimodal content
        images = []
        text_parts = []
        
        for message in request.messages:
            for content in message.content:
                if content.type == "text":
                    text_parts.append(content.text)
                elif content.type == "image":
                    # Load image
                    from sub.vision_processor import VisionProcessor
                    processor = VisionProcessor()
                    image = processor.load_image(content.image)
                    image = processor.smart_resize(image)
                    images.append(image)
        
        # Combine into prompt with vision placeholders
        prompt = "\n".join(text_parts)
        
        # Add vision tokens to prompt
        for i in range(len(images)):
            prompt = f"<|vision_start|><|image_pad|><|vision_end|>{prompt}"
        
        # Process images to tensors
        pixel_values = []
        for img in images:
            tensor = processor.preprocess_image(
                img,
                device=gpt_distr.device,
                dtype=gpt_distr.dtype
            )
            pixel_values.append(tensor)
        
        if pixel_values:
            pixel_values = torch.cat(pixel_values, dim=0)
        else:
            pixel_values = None
        
        # Generate with multimodal inputs
        samples = gpt_distr.start(
            n_samples=request.n_samples,
            tokens_per_sample=request.max_tokens,
            prompt=prompt,
            pixel_values=pixel_values,
        )
        
        return GenerateResponse(samples=samples)
        
    except Exception as e:
        logger.error(f"Error in multimodal generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

### Phase 6: Update Model Preparation Script

Update `src/prepare_model.py`:

```python
# Add support for Qwen2-VL models

def is_vision_model(model_name: str) -> bool:
    """Check if model is a vision-language model."""
    return "VL" in model_name or "vision" in model_name.lower()

def split_vl_model(state_dict: dict, n_nodes: int, model_path: Path):
    """
    Split vision-language model with special handling for vision encoder.
    
    Strategy:
    - Node 0 (starter): embedding + vision_encoder + first N layers
    - Node 1-K (secondary): middle layers
    - Node K+1 (finisher): last N layers + lm_head
    """
    # Separate vision encoder weights
    vision_weights = {
        k: v for k, v in state_dict.items()
        if k.startswith('vision_encoder') or k.startswith('visual')
    }
    
    # Separate language model weights
    lm_weights = {
        k: v for k, v in state_dict.items()
        if not k.startswith('vision_encoder') and not k.startswith('visual')
    }
    
    # Split language model normally
    chunks = split_and_store_with_finisher(lm_weights, n_nodes, model_path)
    
    # Add vision encoder to starter chunk
    starter_chunk_path = model_path / "chunks" / f"{n_nodes}nodes" / "model_starter.pth"
    starter_state = torch.load(starter_chunk_path)
    starter_state.update(vision_weights)
    torch.save(starter_state, starter_chunk_path)
    
    print(f"Added vision encoder to starter node ({len(vision_weights)} parameters)")
    
    return chunks

# Update main() to handle VL models
if is_vision_model(args.MODEL):
    chunks_subfolder = split_vl_model(state_dict, args.n_nodes, model_path)
else:
    chunks_subfolder = split_and_store_with_finisher(state_dict, args.n_nodes, model_path, verb=True)
```

### Phase 7: Update Model Distribution Logic

Update `src/sub/model_dist.py`:

```python
class GPTDistributed:
    def __init__(self, ...):
        # ... existing init code ...
        
        # Check if model is multimodal
        self.is_multimodal = self._check_if_multimodal(ckpt_dir)
        
        if self.is_multimodal:
            logger.info("Detected multimodal (VL) model")
            # Use different model classes
            from sub.model import Qwen2VLConfig, GPTMultimodal
            from sub.submodels import StarterNodeMultimodal
            # ... configure accordingly
    
    def start(
        self,
        n_samples: int,
        tokens_per_sample: int,
        prompt: str,
        pixel_values: Optional[torch.Tensor] = None,  # NEW
        **kwargs
    ):
        """
        Start generation with optional vision input.
        """
        # Tokenize prompt
        tokens = self.tokenizer.encode(prompt, ...)
        
        # If multimodal, prepare vision token masks
        if self.is_multimodal and pixel_values is not None:
            # Find vision placeholder positions in tokens
            vision_token_mask = self._create_vision_mask(tokens)
            
            # Encode images using vision encoder
            # (on starter node only)
            ...
        
        # Continue with existing generation loop
        ...
```

## Testing Strategy

### Test 1: Model Download & Preparation
```bash
cd /Users/madinaalzhanova/Desktop/newmind_internship/FL/MDI-LLM

# Download Qwen2-VL-2B-Instruct
python src/prepare_model.py \
  Qwen/Qwen2-VL-2B-Instruct \
  --ckpt-folder ./src/checkpoints \
  --n-nodes 2 \
  --dtype bfloat16 \
  --skip-convert  # Use HF format directly initially
```

### Test 2: Single-Node Inference (Validation)
```python
# Test script: test_qwen_vl_single.py
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "./src/checkpoints/Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype="auto",
    device_map="cpu"
)

processor = AutoProcessor.from_pretrained(
    "./src/checkpoints/Qwen/Qwen2-VL-2B-Instruct"
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://example.com/test.jpg"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

generated_ids = model.generate(**inputs, max_new_tokens=128)
output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

### Test 3: Distributed Inference
```bash
# Terminal 1: Start secondary node
ENABLE_ENCRYPTION=1 \
python src/secondary.py \
  --chunk ./src/checkpoints/Qwen/Qwen2-VL-2B-Instruct/chunks/2nodes/model_secondary0.pth \
  --nodes-config ./src/settings_distr/config_2nodes_cpu.json 0 \
  --dtype bfloat16

# Terminal 2: Start FastAPI gateway
python src/fastapi_gateway.py \
  --host 0.0.0.0 \
  --port 8000 \
  --ckpt ./src/checkpoints/Qwen/Qwen2-VL-2B-Instruct \
  --nodes-config ./src/settings_distr/config_2nodes_cpu.json \
  --sequence-length 1024

# Terminal 3: Test API
curl -X POST http://localhost:8000/generate/multimodal \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image", "image": "https://example.com/image.jpg"},
        {"type": "text", "text": "What is in this image?"}
      ]
    }],
    "max_tokens": 100
  }'
```

## Challenges & Solutions

### Challenge 1: Vision Encoder Size
**Problem**: Vision encoder is large (~500MB), increases starter node memory.
**Solution**: 
- Quantize vision encoder separately
- Consider keeping vision encoder on separate device if available
- Use smaller image resolutions (adjust min_pixels/max_pixels)

### Challenge 2: Variable Sequence Lengths
**Problem**: Different images → different numbers of visual tokens.
**Solution**:
- Pad visual tokens to maximum within batch
- Use attention masks to ignore padding
- Adjust buffer sizes in socket communication

### Challenge 3: M-ROPE Implementation
**Problem**: Qwen2-VL uses specialized positional embeddings.
**Solution**:
- Use HuggingFace's implementation initially
- Gradually replace with custom implementation for optimization
- Ensure positional info is maintained across nodes

### Challenge 4: Latency
**Problem**: Image encoding adds latency.
**Solution**:
- Cache encoded images when possible
- Preprocess images asynchronously
- Consider streaming for large batches

## Performance Considerations

### Memory Usage
```
Component          | Qwen2-1.7B | Qwen2-VL-2B | Increase
-------------------|------------|-------------|----------
Language Model     | ~3.5 GB    | ~4 GB       | +500 MB
Vision Encoder     | 0          | ~600 MB     | +600 MB
Total (BF16)       | ~3.5 GB    | ~4.6 GB     | +1.1 GB
```

### Throughput
- Vision encoding: ~50-100ms per image (CPU)
- Text generation: similar to text-only model
- Overall: expect ~20-30% slower due to vision processing

## Next Steps

1. **Phase 1** (Week 1): Implement core modules
   - [ ] VisionProcessor class
   - [ ] VisionEncoder in model.py
   - [ ] Update Config classes
   
2. **Phase 2** (Week 2): Distributed modifications
   - [ ] Update submodels.py
   - [ ] Modify model_dist.py
   - [ ] Update prepare_model.py
   
3. **Phase 3** (Week 3): API & Testing
   - [ ] Update fastapi_gateway.py
   - [ ] Write integration tests
   - [ ] Performance benchmarking
   
4. **Phase 4** (Week 4): Optimization
   - [ ] Optimize vision encoder
   - [ ] Improve batching
   - [ ] Add caching

## References

- [Qwen2-VL HuggingFace](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- [Qwen2-VL GitHub](https://github.com/QwenLM/Qwen2-VL)
- [Qwen2-VL Paper](https://arxiv.org/abs/2409.12191)
- [qwen-vl-utils](https://github.com/QwenLM/Qwen-VL/tree/master/qwen-vl-utils)
