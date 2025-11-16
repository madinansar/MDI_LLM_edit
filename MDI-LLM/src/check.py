
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("src/checkpoints/meta-llama/Llama-3.2-1B/starter")
print(m)

