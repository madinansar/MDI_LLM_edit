# Phase 2 Progress - Model Download & Preparation

## Status: IN PROGRESS ⏳

### What's Been Done

1. **✅ Updated `prepare_model.py`**
   - Added `is_vision_model()` function to detect VL models
   - Implemented `split_vl_model()` for special VL splitting
   - Vision encoder weights stay with starter node
   - Language model weights split normally

2. **✅ Created Custom Download Script** (`scripts/download_qwen2_vl.py`)
   - Uses HuggingFace transformers directly (avoids conversion issues)
   - Downloads Qwen2-VL-2B-Instruct
   - Maps HF keys to LitGPT format
   - Splits model for distributed inference
   - **Currently running...**

3. **✅ Updated Config**
   - Added layer configuration for 28-layer VL models (2 nodes)

### Current Task

**Downloading Qwen2-VL-2B-Instruct** (~4.4GB)
```bash
python3.10 scripts/download_qwen2_vl.py --n-nodes 2 --dtype bfloat16
```

Progress:
- model-00001-of-00002.safetensors: 3.99G
- model-00002-of-00002.safetensors: 429M
- **Status: Downloading...**

### Key Implementation Details

#### HF to LitGPT Key Mapping

The script automatically maps HuggingFace Qwen2-VL keys to our LitGPT format:

```python
HF Format                     →  LitGPT Format
─────────────────────────────────────────────────────
visual.*                      →  visual.* (unchanged)
model.embed_tokens.weight     →  transformer.wte.weight
model.layers.N.*              →  transformer.h.N.*
model.norm.weight             →  transformer.ln_f.weight
lm_head.weight                →  lm_head.weight
```

#### VL Model Splitting Strategy

```
┌─────────────────────────────────────┐
│        Full Qwen2-VL Model          │
│  ├── visual.* (vision encoder)      │
│  ├── model.* (language model)       │
│  └── lm_head (output projection)    │
└─────────────────────────────────────┘
              ↓
        Split into 2 nodes
              ↓
┌─────────────────────────┐  ┌────────────────────────┐
│   Starter Node          │  │  Secondary/Finisher    │
│  ├── visual.* (ALL)     │  │  ├── transformer.h.12+ │
│  ├── transformer.wte    │  │  ├── transformer.ln_f  │
│  └── transformer.h.0-11 │  │  └── lm_head           │
└─────────────────────────┘  └────────────────────────┘
```

### Expected Output Structure

After download completes:
```
src/checkpoints/Qwen2-VL-2B-Instruct/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── preprocessor_config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
└── chunks/
    └── 2nodes_finisher/
        ├── model_starter.pth      # Embedding + vision encoder + layers 0-11
        └── model_secondary0.pth   # Layers 12-27 + ln_f + lm_head
```

### Next Steps (After Download)

1. **Verify model chunks**
   - Check file sizes
   - Verify layer counts
   - Test loading

2. **Update model_dist.py**
   - Add multimodal support to GPTDistributed
   - Handle image inputs
   - Pass pixel_values to StarterNode

3. **Test distributed inference**
   - Start secondary node
   - Start FastAPI gateway
   - Send test request with image

### Estimated Timeline

- ⏳ Download: ~10-15 minutes (depending on network speed)
- ⏱️ Verification: 5 minutes
- ⏱️ model_dist.py updates: 20-30 minutes
- ⏱️ Testing: 15-20 minutes

**Total: ~1 hour**

---

## Commands to Run After Download

### 1. Verify Chunks
```bash
ls -lh src/checkpoints/Qwen2-VL-2B-Instruct/chunks/2nodes_finisher/
```

### 2. Check Model Info
```python
python3.10 -c "
import torch
starter = torch.load('src/checkpoints/Qwen2-VL-2B-Instruct/chunks/2nodes_finisher/model_starter.pth', map_location='cpu')
secondary = torch.load('src/checkpoints/Qwen2-VL-2B-Instruct/chunks/2nodes_finisher/model_secondary0.pth', map_location='cpu')

print(f'Starter keys: {len(starter)}')
print(f'Secondary keys: {len(secondary)}')

# Check for vision encoder
vision_keys = [k for k in starter.keys() if 'visual' in k]
print(f'Vision encoder keys: {len(vision_keys)}')
"
```

### 3. Test Model Loading
```bash
python3.10 tests/test_qwen_vl_integration.py
```

---

## Troubleshooting

### If Download Fails
- Check internet connection
- Check HuggingFace Hub access
- Try with `--model-name Qwen/Qwen2-VL-2B` (non-instruct version)
- Check disk space (~5GB needed)

### If Splitting Fails
- Check the error message
- Verify the state dict keys match expected format
- May need to adjust key mapping in `map_hf_to_litgpt_key()`

### If Loading Fails
- Check PyTorch version
- Verify bfloat16 support
- Try with float16 instead

---

## Architecture Details

### Qwen2-VL-2B-Instruct Specifications

**Vision Encoder:**
- Type: ViT (Vision Transformer)
- Patch size: 14×14
- Layers: 32
- Hidden dim: 1536
- Parameters: ~600M

**Language Model:**
- Architecture: Qwen2-based
- Layers: 28
- Hidden dim: 1536
- Heads: 16
- Parameters: ~1.8B

**Total:** ~2.4B parameters

### Memory Requirements

```
Component              | BF16 Size | FP32 Size
-----------------------|-----------|-----------
Vision Encoder         | ~600 MB   | ~1.2 GB
Language Model (full)  | ~3.6 GB   | ~7.2 GB
Language Model (split) | ~1.8 GB   | ~3.6 GB per node

Total (2 nodes):       | ~2.4 GB + ~1.8 GB = ~4.2 GB
```

---

Updated: December 20, 2025
