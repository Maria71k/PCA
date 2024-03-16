from sklearn.decomposition import PCA
import numpy as np

# Sample data
X = np.array([[1, 2], [1, 3], [2, 3], [2, 4]])

# Create and fit PCA model
pca = PCA(n_components=1)
pca.fit(X)

# Transform data
X_transformed = pca.transform(X)
print("Transformed data:", X_transformed)
