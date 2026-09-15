import torch
from transformers import AutoModelForCausalLM

package = AutoModelForCausalLM.from_pretrained("Xenova/llama2.c-stories15M")

emb = package.model.embed_tokens.weight
q0 = package.model.layers[0].self_attn.q_proj.weight
down = package.model.layers[0].mlp.down_proj.weight
norm = package.model.layers[0].post_attention_layernorm.weight

for row, colum in q0:
    print(row, column)

# Code for visualizing the structure of the tensors

# for name, tensor in package.named_parameters():
#     print(name, tensor.shape, tensor.numel())
