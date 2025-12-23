# Quick Start: Testing Your New Components

## Run Integration Tests

```bash
cd /Users/madinaalzhanova/Desktop/newmind_internship/FL/MDI-LLM
source mdi_venv/bin/activate
python3.10 tests/test_qwen_vl_integration.py
```

Expected output: All tests should pass ✓

## Test Individual Components

### 1. Test Vision Processor
```python
python3.10 -c "
import sys
sys.path.insert(0, 'src')
from sub.vision_processor import VisionProcessor
from PIL import Image
import numpy as np
import torch

processor = VisionProcessor()
print('✓ VisionProcessor initialized')

# Create test image
img_array = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
image = Image.fromarray(img_array, 'RGB')

# Process
tensor = processor.preprocess_image(image, torch.device('cpu'))
print(f'✓ Tensor shape: {tensor.shape}')
"
```

### 2. Test Config Loading
```python
python3.10 -c "
import sys
sys.path.insert(0, 'src')
from sub.model import Config

config = Config.from_name('Qwen2-VL-2B-Instruct')
print(f'✓ Config: {config.name}')
print(f'  Multimodal: {config.is_multimodal}')
print(f'  Vision layers: {config.vision_encoder_depth}')
"
```

### 3. Test Vision Encoder
```python
python3.10 -c "
import sys
import torch
sys.path.insert(0, 'src')
from sub.model import Config, VisionEncoder

config = Config.from_name('Qwen2-VL-2B-Instruct')
encoder = VisionEncoder(config)
print('✓ VisionEncoder initialized')

# Test forward pass
dummy_img = torch.randn(1, 3, 224, 224)
output = encoder(dummy_img)
print(f'✓ Output shape: {output.shape}')
"
```

### 4. Test StarterNode
```python
python3.10 -c "
import sys
sys.path.insert(0, 'src')
from sub.model import Config
from sub.submodels import StarterNode

config = Config.from_name('Qwen2-VL-2B-Instruct')
starter = StarterNode(config, n_transf_layers=5, verb=False)
print('✓ StarterNode initialized')
print(f'  Is multimodal: {starter.is_multimodal}')
print(f'  Has vision encoder: {hasattr(starter, \"vision_encoder\")}')
"
```

## Next Steps

### Download Qwen2-VL Model
```bash
# This will be updated in next phase
python src/prepare_model.py \
  Qwen/Qwen2-VL-2B-Instruct \
  --ckpt-folder ./src/checkpoints \
  --n-nodes 2 \
  --dtype bfloat16
```

### Check What's Been Modified
```bash
git status
git diff src/sub/config.py
git diff src/sub/model.py
git diff src/sub/submodels.py
```

## Troubleshooting

### Import Errors
If you get import errors:
```bash
# Ensure you're in the correct directory
cd /Users/madinaalzhanova/Desktop/newmind_internship/FL/MDI-LLM

# Activate virtual environment
source mdi_venv/bin/activate

# Check Python version
python3.10 --version

# Check torch is installed
python3.10 -c "import torch; print(torch.__version__)"
```

### Module Not Found
If `sub` module not found:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))
```

### Test Failures
If tests fail:
1. Check you have all dependencies: `pip list | grep -E "torch|transformers|pillow"`
2. Verify virtual environment is activated
3. Check Python version: should be 3.10
4. Look at error messages for missing imports

## Files You Can Review

1. **Vision Processing:** [src/sub/vision_processor.py](../src/sub/vision_processor.py)
2. **Config Updates:** [src/sub/config.py](../src/sub/config.py) (search for "Qwen2-VL")
3. **Model Changes:** [src/sub/model.py](../src/sub/model.py) (search for "VisionEncoder")
4. **StarterNode Updates:** [src/sub/submodels.py](../src/sub/submodels.py)
5. **Tests:** [tests/test_qwen_vl_integration.py](../tests/test_qwen_vl_integration.py)

## Documentation

- **Full Guide:** [docs/qwen-vl-integration-guide.md](qwen-vl-integration-guide.md)
- **Roadmap:** [docs/qwen-vl-implementation-roadmap.md](qwen-vl-implementation-roadmap.md)
- **Summary:** [docs/implementation-summary.md](implementation-summary.md)
