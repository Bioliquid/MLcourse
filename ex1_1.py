import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt

X = pd.read_csv('37_25.csv', header=None)

pca = PCA(n_components=9, svd_solver='full')

X_transformed = pca.fit(X).transform(X)

print(np.round(X_transformed[0][0], 3))
print(np.round(X_transformed[0][1], 3))

print(np.round(np.cumsum(pca.explained_variance_ratio_), 3)[1])
print(next(x[0] for x in enumerate(np.cumsum(pca.explained_variance_ratio_)) if x[1] > 0.85) + 1)

pca = PCA(n_components=2, svd_solver='full')

X_transformed = pca.fit(X).transform(X)

plt.plot(X_transformed[:101, 0], X_transformed[:101, 1], 'o', markerfacecolor='red', markeredgecolor='k', markersize=8)
plt.show()