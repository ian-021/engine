import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Xenova/llama2.c-stories15M")

for name, tensor in model.named_parameters():
    print(name, tensor.shape, tensor.numel())
