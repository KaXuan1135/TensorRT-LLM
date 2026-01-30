import torch
from safetensors.torch import load_file, save_file, save_model

weights = load_file("model.safetensors")
if "lm_head.weight" not in weights:
    print("Fixing missing lm_head.weight...")
    weights["lm_head.weight"] = weights["model.embed_tokens.weight"].clone().detach()
    save_file(weights, "model.safetensors")
    print("Done! Try running the conversion again.")