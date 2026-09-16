import torch
import struct
import numpy as np
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Xenova/llama2.c-stories15M")

hidden_size = model.config.hidden_size
intermediate_size = model.config.intermediate_size
vocab_size = model.config.vocab_size
num_hidden_layers = model.config.num_hidden_layers

# HEADER ORDER (C struct must match):
#   hidden_size, intermediate_size, vocab_size, num_hidden_layers
header = struct.pack("<4i", hidden_size, intermediate_size, vocab_size, num_hidden_layers)


def write_tensor(f, t):
    # tensor -> flat float32 bytes, appended to the file
    assert t.dtype == torch.float32, f"expected float32, got {t.dtype}"
    arr = t.detach().flatten().numpy()
    f.write(arr.tobytes())


# BODY ORDER (C reader must walk the same order):
#   embed_tokens
#   for each layer: q, k, v, o, gate, up, down, input_layernorm, post_attention_layernorm
#   norm
with open("stories15M.bin", "wb") as f:
    f.write(header)

    write_tensor(f, model.model.embed_tokens.weight)

    for n in range(num_hidden_layers):
        layer = model.model.layers[n]
        write_tensor(f, layer.self_attn.q_proj.weight)
        write_tensor(f, layer.self_attn.k_proj.weight)
        write_tensor(f, layer.self_attn.v_proj.weight)
        write_tensor(f, layer.self_attn.o_proj.weight)
        write_tensor(f, layer.mlp.gate_proj.weight)
        write_tensor(f, layer.mlp.up_proj.weight)
        write_tensor(f, layer.mlp.down_proj.weight)
        write_tensor(f, layer.input_layernorm.weight)
        write_tensor(f, layer.post_attention_layernorm.weight)

    write_tensor(f, model.model.norm.weight)

# reference values for the C-side check
print("embed row 42, first 5:", model.model.embed_tokens.weight[42, :5].tolist())
