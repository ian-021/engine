import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Xenova/llama2.c-stories15M")

print(model.config.hidden_size)
