import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt

scores = np.genfromtxt('X_reduced_792.csv', delimiter=';')
loadings = np.genfromtxt('X_loadings_792.csv', delimiter=';')

values = np.dot(scores,loadings.T)

plt.imshow(values, cmap='Greys_r')
plt.show()