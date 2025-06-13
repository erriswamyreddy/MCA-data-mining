# Aim: Implement Principal Component Analysis (PCA) for Dimensionality Reduction (Python Code using sklearn)

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import datasets

# Step 1: Load the dataset
iris = datasets.load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Step 2: Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Step 3: Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_scaled)

# Step 4: Print explained variance ratio
explained_variance = pca.explained_variance_ratio_
print("Cumulative Explained Variance:", np.sum(explained_variance))

# Step 5: Visualize the results
plt.figure(figsize=(8, 6))
for target, color in zip(np.unique(iris.target), ['r', 'g', 'b']):
    plt.scatter(pca_result[iris.target == target, 0],
                pca_result[iris.target == target, 1],
                label=iris.target_names[target],
                alpha=0.7,
                edgecolors='k')

plt.title("PCA on Iris Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.show()
