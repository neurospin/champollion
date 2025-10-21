import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from sklearn.decomposition import PCA
# Step 1: Define the base path
#base_path = '/neurospin/dico/jlaval/Output/5_models_orbital_3_layer_proj/pepper_05_*/right_21045_UKB_random_embeddings/full_embeddings.csv'
#base_path = '/neurospin/dico/jlaval/Output/5_models_FIP_right_3_layer_proj/ResNet_*/right_21043_UKB_random_embeddings/full_embeddings.csv'
#base_path = '/neurospin/dico/jlaval/Output/5_models_FIP_right_3_layer_proj/pepper_05_*/three_datasets_UKB_random_embeddings/full_embeddings.csv'
base_path = '/neurospin/dico/data/deep_folding/current/models/Champollion_V0/FIP_right/20-16-33_*/three_datasets_UKB_random_epoch*_embeddings/full_embeddings.csv'
# Step 2: Use glob to find all matching files
file_paths = glob.glob(base_path)
# Step 3: Load and concatenate all embeddings into a single DataFrame
embeddings_list = [pd.read_csv(file, index_col=0) for file in file_paths]
for embeddings_UKB, file in zip(embeddings_list,file_paths):
    print("\n", file)
    # Print basic statistics
    print("\n", embeddings_UKB.describe(), "\n")
    print("Minimum std among the dimensions:")
    print(embeddings_UKB.std(axis=0).min(), "\n")
    print("Maximum std among the dimensions:")
    print(embeddings_UKB.std(axis=0).max(), "\n")
    # Step 4: Perform PCA
    n_components = 32
    pca = PCA(n_components=n_components)
    pca.fit(embeddings_UKB)
    # Print explained variance ratio
    print("Explained variance ratio for each PC:")
    print(pca.explained_variance_ratio_, "\n")
    # Number of components to explain 99% of the variance
    print((np.cumsum(pca.explained_variance_ratio_) < 0.99).sum())
    # Step 5: Visualize explained variance
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, n_components + 1), np.cumsum(pca.explained_variance_ratio_))
    plt.grid(True)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA on the Combined Embeddings')
    plt.show()