
import torch

CHECKPOINT_PATH = "checkpoints/pythonmind_config_c.pt"

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True
)

print("\n========== CONFIG C CHECKPOINT ==========")

print("Epoch:", checkpoint["epoch"])
print("Saved Validation Loss:", checkpoint["validation_loss"])
print("Saved Validation Accuracy:", checkpoint["validation_accuracy"])

print("Training Functions:", checkpoint["training_functions"])
print("Validation Functions:", checkpoint["validation_functions"])

print("Learning Rate:", checkpoint["learning_rate"])
print("Weight Decay:", checkpoint["weight_decay"])

print("=========================================")
