import os
import json
import re
import matplotlib.pyplot as plt

# Directory where your logs are saved
log_dir = "/neurospin/dico/babdelghani/Runs/02_champollion_v1/Program/2023_jlaval_STSbabies/" 

pattern = re.compile(r"SOr_left_UKB40_batch_size_(\d+)_logs_lightning_logs_version_\d+")


plt.figure(figsize=(12, 6))

for filename in os.listdir(log_dir):
    match = pattern.match(filename)
    if match:
        batch_size = match.group(1)
        label = f"Batch Size {batch_size}"
        filepath = os.path.join(log_dir, filename)
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            epochs = [entry[1] for entry in data]
            losses = [entry[2] for entry in data]
            plt.plot(epochs, losses, label=label)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

plt.title("Training Loss vs Epochs for Different Batch Sizes")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

output_path = os.path.join(log_dir, "loss_plot.png")
plt.savefig(output_path)
print(f"Plot saved as: {output_path}")

