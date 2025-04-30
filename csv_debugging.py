import pandas as pd

# Load your CSV
df = pd.read_csv("/neurospin/dico/data/deep_folding/current/datasets/UkBioBank40/participants_sex_age_volumes.csv")

# Count NaNs per column
nan_counts = df.isna().sum()

# Print it nicely
print("NaNs per column:")
print(nan_counts)
