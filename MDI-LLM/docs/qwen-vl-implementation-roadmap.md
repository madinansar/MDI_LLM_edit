# Qwen2-VL Integration - Implementation Roadmap

## Quick Start: Minimal Viable Implementation

If you want to get something working quickly, here's the minimal path:

### Option A: Use HuggingFace Transformers Directly (Fastest - 1-2 days)

**Pros**: Minimal code changes, proven implementation  
**Cons**: Less control, harder to optimize for distributed inference

```python
# In fastapi_gateway.py, add a separate endpoint for VL
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Load VL model separately (not distributed)
vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype="auto",
    device_map="cpu"  # or "auto" for GPU
)
vl_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

@app.post("/generate/vision")
async def generate_vision(request: MultimodalRequest):
    # Use HF implementation directly
    messages = request.messages
    text = vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = vl_processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    generated_ids = vl_model.generate(**inputs, max_new_tokens=request.max_tokens)
    output = vl_processor.batch_decode(generated_ids, skip_special_tokens=True)
    return {"output": output}
```

**Use this approach if**: You need something working immediately for testing/demos.

---

### Option B: Full Distributed Integration (Recommended - 2-4 weeks)

This properly integrates Qwen2-VL into your existing MDI-LLM architecture.

## Phase-by-Phase Implementation

### Phase 1: Foundation (Priority: HIGH, Time: 3-4 days)

#### 1.1 Update Dependencies
```bash
cd /Users/madinaalzhanova/Desktop/newmind_internship/FL/MDI-LLM
source mdi_venv/bin/activate

# Upgrade transformers
pip install --upgrade transformers>=4.45.0
pip install qwen-vl-utils
pip install pillow>=10.0.0
pip install torchvision

# Update requirements.txt
echo "transformers>=4.45.0" >> requirements.txt
echo "qwen-vl-utils>=0.0.2" >> requirements.txt
echo "pillow>=10.0.0" >> requirements.txt
echo "torchvision>=0.19.0" >> requirements.txt
```

#### 1.2 Create Vision Processor Module
**File**: `src/sub/vision_processor.py`  
**Lines of Code**: ~200  
**Dependencies**: PIL, requests, base64

This module handles:
- Loading images from URLs, files, base64
- Smart resizing (maintain aspect ratio, patch-size alignment)
- Converting to tensors with proper normalization

See full implementation in the integration guide.

#### 1.3 Test Vision Processor Standalone
```python
# test_vision_processor.py
from sub.vision_processor import VisionProcessor
import torch

processor = VisionProcessor()

# Test URL loading
image = processor.load_image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg")
print(f"Loaded image: {image.size}")

# Test resizing
resized = processor.smart_resize(image, min_pixels=256*28*28, max_pixels=512*28*28)
print(f"Resized to: {resized.size}")

# Test preprocessing
tensor = processor.preprocess_image(resized, device=torch.device('cpu'))
print(f"Tensor shape: {tensor.shape}")
```

---

### Phase 2: Model Architecture (Priority: HIGH, Time: 4-5 days)

#### 2.1 Download & Inspect Qwen2-VL Model
```bash
# First, download the model to inspect its structure
python -c "
from transformers import Qwen2VLForConditionalGeneration
model = Qwen2VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2-VL-2B-Instruct',
    torch_dtype='auto',
    device_map='cpu'
)
print('Model structure:')
print(model)
print('\nState dict keys:')
for key in list(model.state_dict().keys())[:20]:
    print(f'  {key}: {model.state_dict()[key].shape}')
"
```

#### 2.2 Add Qwen2VLConfig to config.py
**File**: `src/sub/config.py`  
**Action**: Add Qwen2-VL-2B-Instruct configuration

```python
# Add to configs list in config.py
{
    "name": "Qwen/Qwen2-VL-2B-Instruct",
    "hf_config": {
        "org": "Qwen",
        "name": "Qwen2-VL-2B-Instruct",
    },
    "scale_embeddings": False,
    "block_size": 32768,
    "vocab_size": 151936,
    "n_layer": 28,
    "n_head": 16,
    "head_size": 128,
    "n_embd": 1536,
    "rotary_percentage": 1.0,
    "parallel_residual": False,
    "bias": False,
    "norm_class_name": "RMSNorm",
    "mlp_class_name": "LLaMAMLP",
    "intermediate_size": 8960,
    "rope_base": 1000000,
    "n_query_groups": 16,
    # VL-specific
    "vision_encoder_embed_dim": 1536,
    "vision_encoder_depth": 32,
    "vision_encoder_num_heads": 16,
    "vision_patch_size": 14,  # Qwen2-VL uses 14x14 patches
    "vision_start_token_id": 151652,
    "vision_end_token_id": 151653,
    "image_token_id": 151655,
    "use_mrope": True,
}
```

#### 2.3 Extend Config Class
**File**: `src/sub/model.py`  
**Action**: Add vision-specific config fields

```python
@dataclass
class Config:
    # ... existing fields ...
    
    # Vision-Language Model fields
    vision_encoder_embed_dim: Optional[int] = None
    vision_encoder_depth: Optional[int] = None
    vision_encoder_num_heads: Optional[int] = None
    vision_patch_size: Optional[int] = None
    vision_start_token_id: Optional[int] = None
    vision_end_token_id: Optional[int] = None
    image_token_id: Optional[int] = None
    use_mrope: bool = False
    
    @property
    def is_multimodal(self) -> bool:
        """Check if this is a vision-language model."""
        return self.vision_encoder_embed_dim is not None
```

#### 2.4 Add Vision Encoder Class
**File**: `src/sub/model.py`  
**Lines**: ~150  
**Key Methods**: 
- `__init__`: Initialize ViT-based encoder
- `forward`: Process images → visual embeddings

**Important**: Initially use a simplified vision encoder. You can load the actual Qwen2-VL vision weights later.

```python
class VisionEncoder(nn.Module):
    """Simplified Vision Transformer for Qwen2-VL."""
    
    def __init__(self, config: Config):
        super().__init__()
        # See full implementation in integration guide
        pass
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [B, 3, H, W]
        # returns: [B, num_patches, embed_dim]
        pass
```

---

### Phase 3: Distributed Nodes (Priority: MEDIUM, Time: 5-6 days)

#### 3.1 Extend StarterNode for Multimodal
**File**: `src/sub/submodels.py`

Key changes:
1. Add `vision_encoder` to StarterNode
2. Modify `forward()` to accept `pixel_values`
3. Handle vision token insertion
4. Update communication protocol to include vision features

```python
class StarterNode(NodePrototype):
    def __init__(self, config: Config, n_transformer_layers: int, **kwargs):
        super().__init__(**kwargs)
        
        # Existing embeddings
        self.transformer.wte = nn.Embedding(config.padded_vocab_size, config.n_embd)
        
        # Add vision encoder if multimodal
        if config.is_multimodal:
            self.vision_encoder = VisionEncoder(config)
            self.vision_projection = nn.Linear(
                config.vision_encoder_embed_dim,
                config.n_embd
            )
        
        # Rest of init...
```

#### 3.2 Update Model Distribution Logic
**File**: `src/sub/model_dist.py`

Changes:
- Detect if model is multimodal from config
- Load vision encoder weights into starter node
- Update `start()` method to accept images

```python
class GPTDistributed:
    def __init__(self, ...):
        # ... existing code ...
        
        # Check if multimodal
        if self.model_config.is_multimodal:
            logger.info("Initializing multimodal (VL) model")
            # Different initialization for VL models
    
    def start(
        self,
        prompt: str,
        n_samples: int = 1,
        tokens_per_sample: int = 100,
        images: Optional[List] = None,  # NEW
        **kwargs
    ):
        # Handle image preprocessing
        # Encode images if provided
        # Continue with text generation
        pass
```

---

### Phase 4: Model Preparation (Priority: MEDIUM, Time: 2-3 days)

#### 4.1 Update prepare_model.py

Add logic to:
1. Detect VL models
2. Separate vision encoder weights
3. Distribute language model layers
4. Keep vision encoder in starter chunk

```python
def split_vl_model(state_dict: dict, n_nodes: int, model_path: Path):
    """Special splitting logic for VL models."""
    
    # Identify vision encoder keys
    vision_keys = [k for k in state_dict.keys() if 'visual' in k or 'vision' in k]
    
    # Separate weights
    vision_weights = {k: state_dict[k] for k in vision_keys}
    lm_weights = {k: v for k, v in state_dict.items() if k not in vision_keys}
    
    # Split language model normally
    chunks = split_and_store_with_finisher(lm_weights, n_nodes, model_path)
    
    # Add vision encoder to starter
    starter_path = chunks['starter']
    starter_state = torch.load(starter_path)
    starter_state.update(vision_weights)
    torch.save(starter_state, starter_path)
    
    return chunks
```

#### 4.2 Prepare Qwen2-VL-2B-Instruct
```bash
python src/prepare_model.py \
  Qwen/Qwen2-VL-2B-Instruct \
  --ckpt-folder ./src/checkpoints \
  --n-nodes 2 \
  --dtype bfloat16
```

---

### Phase 5: API Gateway (Priority: LOW, Time: 2-3 days)

#### 5.1 Add Multimodal Endpoint
**File**: `src/fastapi_gateway.py`

```python
from pydantic import BaseModel
from typing import List, Optional
from fastapi import File, UploadFile

class ContentItem(BaseModel):
    type: str  # "text" or "image"
    text: Optional[str] = None
    image: Optional[str] = None  # URL or base64

class Message(BaseModel):
    role: str = "user"
    content: List[ContentItem]

class MultimodalRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = 100
    n_samples: int = 1

@app.post("/generate/multimodal")
async def generate_multimodal(request: MultimodalRequest):
    """Generate from text + images."""
    
    # Extract images and text
    images = []
    text_parts = []
    
    for message in request.messages:
        for item in message.content:
            if item.type == "text":
                text_parts.append(item.text)
            elif item.type == "image":
                # Load and preprocess image
                from sub.vision_processor import VisionProcessor
                processor = VisionProcessor()
                img = processor.load_image(item.image)
                img = processor.smart_resize(img)
                images.append(img)
    
    # Build prompt with vision tokens
    prompt = " ".join(text_parts)
    if images:
        # Add vision placeholder tokens
        prompt = f"<|vision_start|><|image_pad|><|vision_end|> {prompt}"
    
    # Call distributed model
    output = gpt_distr.start(
        prompt=prompt,
        n_samples=request.n_samples,
        tokens_per_sample=request.max_tokens,
        images=images,  # Pass images
    )
    
    return {"samples": output}
```

---

### Phase 6: Testing & Validation (Priority: HIGH, Time: 3-4 days)

#### 6.1 Unit Tests

Create `tests/test_vision_processor.py`:
```python
import pytest
from sub.vision_processor import VisionProcessor

def test_load_image_from_url():
    processor = VisionProcessor()
    image = processor.load_image("https://example.com/test.jpg")
    assert image is not None
    assert image.mode == "RGB"

def test_smart_resize():
    processor = VisionProcessor()
    # Test with various sizes
    pass

def test_preprocess():
    # Test tensor conversion
    pass
```

Create `tests/test_vl_model.py`:
```python
def test_vision_encoder():
    from sub.model import VisionEncoder, Config
    config = Config.from_name("Qwen/Qwen2-VL-2B-Instruct")
    encoder = VisionEncoder(config)
    
    # Dummy input
    x = torch.randn(1, 3, 224, 224)
    out = encoder(x)
    
    assert out.shape[0] == 1
    assert out.shape[2] == config.vision_encoder_embed_dim

def test_multimodal_forward():
    # Test full forward pass with images
    pass
```

#### 6.2 Integration Test

Create `tests/test_distributed_vl.py`:
```bash
# Start secondary node in background
python src/secondary.py \
  --chunk ./src/checkpoints/Qwen/Qwen2-VL-2B-Instruct/chunks/2nodes/model_secondary0.pth \
  --nodes-config ./src/settings_distr/config_2nodes_cpu.json 0 \
  --dtype bfloat16 &

# Test with FastAPI
python src/fastapi_gateway.py \
  --ckpt ./src/checkpoints/Qwen/Qwen2-VL-2B-Instruct \
  --nodes-config ./src/settings_distr/config_2nodes_cpu.json &

# Send test request
curl -X POST http://localhost:8000/generate/multimodal \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image", "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"},
        {"type": "text", "text": "What animal is in this image?"}
      ]
    }],
    "max_tokens": 50
  }'
```

---

## Implementation Priority Matrix

| Task | Priority | Effort | Blockers | Can Start |
|------|----------|--------|----------|-----------|
| 1.1 Update dependencies | HIGH | 1h | None | ✅ Now |
| 1.2 Vision processor | HIGH | 1d | 1.1 | ✅ Now |
| 2.1 Download model | HIGH | 2h | 1.1 | ✅ Now |
| 2.2 Add config | HIGH | 2h | 2.1 | After 2.1 |
| 2.3 Extend Config | HIGH | 3h | 2.2 | After 2.2 |
| 2.4 Vision Encoder | HIGH | 2d | 2.3 | After 2.3 |
| 3.1 Update StarterNode | MEDIUM | 2d | 2.4 | After 2.4 |
| 3.2 Update model_dist | MEDIUM | 2d | 3.1 | After 3.1 |
| 4.1 Update prepare_model | MEDIUM | 1d | 2.4 | After 2.4 |
| 4.2 Prepare model | MEDIUM | 1h | 4.1 | After 4.1 |
| 5.1 API endpoint | LOW | 2d | 3.2 | After 3.2 |
| 6.1 Unit tests | HIGH | 2d | Various | Parallel |
| 6.2 Integration test | HIGH | 1d | All above | Final |

---

## Quick Decision Tree

**Question 1**: Do you need distributed inference for VL models?
- **YES** → Follow Option B (Full Integration)
- **NO** → Use Option A (HuggingFace Direct)

**Question 2**: How much time do you have?
- **1-2 days** → Option A
- **2-4 weeks** → Option B

**Question 3**: What's your primary goal?
- **Demo/Prototype** → Option A, then migrate to B
- **Production system** → Option B from start
- **Research/Learning** → Option B (more control)

---

## Common Pitfalls & Solutions

### Pitfall 1: Memory Issues
**Problem**: VL model is larger than text-only  
**Solution**: 
- Use bfloat16 instead of float32
- Increase swap space
- Use smaller image resolutions
- Split across more nodes

### Pitfall 2: Vision Encoder Compatibility
**Problem**: HuggingFace weights don't match custom architecture  
**Solution**:
- Start with HuggingFace's implementation
- Gradually replace components
- Use `state_dict` key mapping

### Pitfall 3: Token Alignment
**Problem**: Vision tokens not properly aligned with text  
**Solution**:
- Use Qwen2-VL's tokenizer directly
- Follow their preprocessing exactly
- Test with simple examples first

### Pitfall 4: Communication Overhead
**Problem**: Sending vision features between nodes is slow  
**Solution**:
- Compress vision features
- Use efficient serialization (e.g., torch.save with compression)
- Consider keeping vision encoder co-located with first few layers

---

## Recommended Approach

**Week 1**: Foundation (Phases 1-2)
- Set up environment
- Implement vision processor
- Add model architecture components
- Test standalone

**Week 2**: Distribution (Phases 3-4)
- Update distributed nodes
- Implement model splitting
- Test with dummy data

**Week 3**: Integration (Phase 5)
- Add API endpoints
- End-to-end testing
- Fix bugs

**Week 4**: Optimization & Polish
- Performance tuning
- Documentation
- Production readiness

---

## Success Metrics

- [ ] Vision processor can load/resize images from various sources
- [ ] Vision encoder produces embeddings of correct shape
- [ ] Starter node accepts both text and image inputs
- [ ] Model splits correctly (vision encoder in starter)
- [ ] Distributed inference produces coherent outputs
- [ ] API accepts multimodal requests
- [ ] Latency < 2x text-only model
- [ ] Memory usage fits on target devices

---

## Getting Help

If you get stuck:

1. **Check existing implementations**:
   - HuggingFace Qwen2-VL code
   - LitGPT multimodal examples
   
2. **Test incrementally**:
   - Single components first
   - Then integration
   
3. **Use logging**:
   - Print tensor shapes at each step
   - Validate intermediate outputs

4. **Fallback strategy**:
   - If distributed VL is too complex initially
   - Start with Option A (direct HF)
   - Gradually migrate components

Would you like me to start implementing any specific component?
