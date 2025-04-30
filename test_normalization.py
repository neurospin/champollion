import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
csv_path = "/neurospin/dico/data/deep_folding/current/datasets/UkBioBank40/participants_sex_age_volumes.csv"
df = pd.read_csv(csv_path)

# Drop NaNs
volumes = df['volume'].dropna()

# Original stats
print(f"Number of samples: {len(volumes)}")
print(f"Min: {volumes.min():.4f}")
print(f"Max: {volumes.max():.4f}")
print(f"Mean: {volumes.mean():.4f}")
print(f"Std Dev: {volumes.std():.4f}")

# Z-score normalization
mean_volume = volumes.mean()
std_volume = volumes.std()
df['volume'] = (df['volume'] - mean_volume) / std_volume

# Save normalized CSV
new_csv_path = "/neurospin/dico/data/deep_folding/current/datasets/UkBioBank40/participants_sex_age_volumes_normalized.csv"
df.to_csv(new_csv_path, index=False)

# Post-normalization stats
print(f"Min after z-scoring: {df['volume'].min():.4f}")
print(f"Max after z-scoring: {df['volume'].max():.4f}")
print(f"Mean after z-scoring: {df['volume'].mean():.4f}")
print(f"Std Dev after z-scoring: {df['volume'].std():.4f}")

# Plot distributions
plt.figure(figsize=(10, 5))

# Before normalization
plt.subplot(1, 2, 1)
plt.hist(volumes, bins=30, color='skyblue', edgecolor='black')
plt.title("Original Volume Distribution")
plt.xlabel("Volume")
plt.ylabel("Count")

# After normalization
plt.subplot(1, 2, 2)
plt.hist(df['volume'], bins=30, color='salmon', edgecolor='black')
plt.title("Z-scored Volume Distribution")
plt.xlabel("Z-scored Volume")
plt.ylabel("Count")

# Save plot
plt.tight_layout()
fig_path = "/neurospin/dico/babdelghani/Runs/02_champollion_v1/Program/2023_jlaval_STSbabies/fig.png"
plt.savefig(fig_path)
plt.close()

print(f"Distribution plot saved to: {fig_path}")
