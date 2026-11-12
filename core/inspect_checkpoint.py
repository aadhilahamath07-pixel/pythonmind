
import torch

CHECKPOINT_PATH = "checkpoints/pythonmind_fullseq_core.pt"

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True
)

print("========== CHECKPOINT INSPECTION ==========")

print("Available Keys:")
print(checkpoint.keys())

print("\nModel Configuration:")
for key in [
    "vocab_size",
    "embedding_dim",
    "num_heads",
    "num_layers",
    "max_seq_length"
]:
    print(f"{key}: {checkpoint.get(key)}")

print("\nTraining Information:")
for key in checkpoint:
    if key not in ["model_state_dict", "optimizer_state_dict"]:
        print(f"{key}: {checkpoint[key]}")

print("\nModel State:")
state = checkpoint["model_state_dict"]

print("Number of tensors:", len(state))

for name, tensor in state.items():
    print(
        f"{name}: shape={tuple(tensor.shape)}, "
        f"mean={tensor.float().mean().item():.6f}, "
        f"std={tensor.float().std().item():.6f}"
    )
