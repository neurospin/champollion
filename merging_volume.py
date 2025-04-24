# import pandas as pd

# path_age_sex = "/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/participants_sex_age.csv"
# path_volumes = "/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/brain_segmentation_volume_l.csv"
# output_path = "/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/partic_sex_age_volumes.csv"

# df_age_sex = pd.read_csv(path_age_sex)  
# df_volumes = pd.read_csv(path_volumes)           
# merged_df = pd.merge(df_age_sex, df_volumes, on="participant_id", how="left")
# merged_df.to_csv(output_path, index=False)

# print(f"Merged file saved at: {output_path}")

# import pandas as pd

# df = pd.read_csv("/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/partic_sex_age_volumes.csv")

# print(df.head(20))  # Show the first few rows

# import pandas as pd

# # Path to your CSV without headers
# input_path = "/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/brain_segmentation_volume.csv"
# output_path = "/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/brain_segmentation_volume_l.csv"
# # Load CSV without headers
# df = pd.read_csv(input_path, header=None)

# # Assign new column names
# df.columns = ["participant_id", "volume"]

# # Save it back to CSV (with new headers)
# df.to_csv(output_path, index=False)

# print("Column labels added and file updated!")

import os

# Paths
old_name = "/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/partic_sex_age_volumes.csv"
new_name = "/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/participants_sex_age_volumes.csv"

# Rename
os.rename(old_name, new_name)

print("✅ File renamed!")

