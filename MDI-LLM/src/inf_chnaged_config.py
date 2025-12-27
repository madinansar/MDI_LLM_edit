# No KV caches
import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoConfig, Qwen2VLForConditionalGeneration
from pathlib import Path
from PIL import Image
import gc

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = Path("./src/checkpoints/Qwen/Qwen2-VL-2B-Instruct")
CHUNKS_DIR = BASE_DIR / "chunks"

STARTER_PATH = CHUNKS_DIR / "model_starter.pth"
SECONDARY_PATH = CHUNKS_DIR / "model_secondary0.pth"

DEVICE = "cuda:0"
DTYPE = torch.float16

# The split point: Starter has 16 layers (0-15), Secondary has 12 (16-27)
STARTER_LAYERS = 16
SECONDARY_LAYERS = 12 

# ============================================================================
# 1. STARTER NODE (Layers 0 -> 15)
# ============================================================================
class StarterNode(nn.Module):
    def __init__(self, device):
        super().__init__()
        print(f"[{device}] Initializing Optimized Starter Node ({STARTER_LAYERS} Layers)...")
        
        # 1. Load & Modify Config
        # We tell the model it ONLY has 16 layers. It saves RAM immediately.
        config = AutoConfig.from_pretrained(BASE_DIR, local_files_only=True)
        config.num_hidden_layers = STARTER_LAYERS
        
        # 2. Init Model Shell (Creates Vision + Embeds + Layers 0-15)
        self.model = Qwen2VLForConditionalGeneration(config)
        
        # 3. CRITICAL FIX: Replace Norm with Identity
        # Even a partial model creates a final Norm layer. We must disable it
        # so we pass RAW hidden states to the next node.
        self.model.model.language_model.norm = nn.Identity()
        
        # 4. Remove Head (Unused in Starter)
        del self.model.lm_head
        gc.collect()
        
        # 5. Load Weights
        print(f"  - Loading {STARTER_PATH.name}...")
        ckpt = torch.load(STARTER_PATH, map_location="cpu")
        state_dict = ckpt['state_dict']
        
        # Map keys from your file format to the model's internal structure
        model_dict = {}
        for k, v in state_dict.items():
            if k.startswith("visual"):
                model_dict[f"model.{k}"] = v
            elif k.startswith("embed_tokens"):
                model_dict[f"model.language_model.{k}"] = v
            elif k.startswith("layers"):
                model_dict[f"model.language_model.{k}"] = v
            else:
                model_dict[k] = v
                
        self.model.load_state_dict(model_dict, strict=False)
        del state_dict, ckpt, model_dict
        
        self.model.to(device, dtype=DTYPE)
        self.model.eval()

    def forward(self, input_ids, pixel_values, image_grid_thw, attention_mask):
        with torch.no_grad():
            outputs = self.model.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        # Return the last hidden state (Layer 15 output)
        return outputs.last_hidden_state


# ============================================================================
# 2. SECONDARY NODE (Layers 16 -> 27)
# ============================================================================
class SecondaryNode(nn.Module):
    def __init__(self, device):
        super().__init__()
        print(f"[{device}] Initializing Optimized Secondary Node ({SECONDARY_LAYERS} Layers)...")
        
        # 1. Load & Modify Config
        # We tell the model it has 12 layers.
        # This creates layers labeled 0 to 11 internally.
        config = AutoConfig.from_pretrained(BASE_DIR, local_files_only=True)
        config.num_hidden_layers = SECONDARY_LAYERS
        
        # 2. Init Model Shell
        self.model = Qwen2VLForConditionalGeneration(config)
        
        # 3. Delete Unused Components
        # Even with fewer layers, Qwen still creates Vision/Embeds. Delete them.
        del self.model.model.visual
        del self.model.model.language_model.embed_tokens
        gc.collect()
        
        # 4. Load Weights
        print(f"  - Loading {SECONDARY_PATH.name}...")
        ckpt = torch.load(SECONDARY_PATH, map_location="cpu")
        state_dict = ckpt['state_dict']
        
        model_dict = {}
        for k, v in state_dict.items():
            # Your split script saved these as layers.0, layers.1... (relative index)
            # This PERFECTLY matches our new 12-layer config (layers.0, layers.1...)
            if k.startswith("layers"):
                model_dict[f"model.language_model.{k}"] = v
            elif k.startswith("norm"):
                model_dict[f"model.language_model.{k}"] = v
            elif k.startswith("lm_head"):
                model_dict[k] = v
                
        self.model.load_state_dict(model_dict, strict=False)
        del state_dict, ckpt, model_dict
        
        self.model.to(device, dtype=DTYPE)
        self.model.eval()

    def forward(self, hidden_states, image_grid_thw, attention_mask):
        """
        We inject hidden_states into 'inputs_embeds'.
        We pass 'pixel_values=None' to ensure the deleted Vision Tower isn't called.
        We pass 'image_grid_thw' so the RoPE layers know how to handle image positions.
        """
        with torch.no_grad():
            outputs = self.model(
                inputs_embeds=hidden_states,   # Input from Starter
                pixel_values=None,             # Disable Vision
                image_grid_thw=image_grid_thw, # Enable RoPE
                attention_mask=attention_mask
            )
        return outputs.logits


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("Loading Processor...")
    # Included fix_mistral_regex=True to suppress your warning
    processor = AutoProcessor.from_pretrained(BASE_DIR, local_files_only=True, fix_mistral_regex=True)
    
    starter = StarterNode(DEVICE)
    secondary = SecondaryNode(DEVICE)
    
    print("\nProcessing Input...")
    image = Image.open("./src/catdog.jpg").convert("RGB")
    
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Describe this image."}]}],
        tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True
    ).to(DEVICE)
    
    print("\nGenerating...")
    current_ids = inputs.input_ids
    current_mask = inputs.attention_mask
    
    # Cache these for reuse
    p_values = inputs.pixel_values
    g_thw = inputs.image_grid_thw

    with torch.no_grad():
        for i in range(50): # Generate 50 tokens
            
            # --- STEP 1: STARTER (Vision + Layers 0-15) ---
            hidden_mid = starter(
                input_ids=current_ids, 
                pixel_values=p_values, 
                image_grid_thw=g_thw, 
                attention_mask=current_mask
            )
            
            # --- STEP 2: SECONDARY (Layers 16-27 + Head) ---
            logits = secondary(
                hidden_states=hidden_mid,
                image_grid_thw=g_thw,
                attention_mask=current_mask
            )
            
            # --- STEP 3: DECODE ---
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            token_str = processor.decode(next_token[0])
            print(token_str, end="", flush=True)
            
            # --- STEP 4: UPDATE ---
            current_ids = torch.cat([current_ids, next_token], dim=1)
            current_mask = torch.cat([current_mask, torch.ones((1, 1), device=DEVICE)], dim=1)

    print("\n\nDone.")

if __name__ == "__main__":
    main()


#     Loading Processor...
# [cuda:0] Initializing Optimized Starter Node (16 Layers)...
#   - Loading model_starter.pth...
# [cuda:0] Initializing Optimized Secondary Node (12 Layers)...
#   - Loading model_secondary0.pth...

# Processing Input...

# Generating...
# The image shows a playful scene featuring a golden retriever puppy and a ginger kitten. The puppy is lying on its side, with its front paws stretched out in front of
