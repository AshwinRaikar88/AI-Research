import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
irisModule = load_iris()
dataset = np.array(irisModule.data)
print(dataset)

covarianceMatrix = pd.DataFrame(data = np.cov(dataset, rowvar = False), columns = irisModule.feature_names,
                                index = irisModule.feature_names)
print(covarianceMatrix)

eigenvalues, eigenvectors = np.linalg.eig(covarianceMatrix)
print(eigenvalues)
print('\n')
print(eigenvectors)

totalSum = sum(eigenvalues)
variablesExplained = [(i / totalSum) for i in sorted(eigenvalues, reverse = True)]
import matplotlib.pyplot as plt
plt.pie(variablesExplained)
plt.show()
print(variablesExplained)

featureVector = eigenvectors[:,:2]
print(featureVector)

featureVectorTranspose = np.transpose(featureVector)
datasetTranspose = np.transpose(dataset)
newDatasetTranspose = np.matmul(featureVectorTranspose, datasetTranspose)
newDataset = np.transpose(newDatasetTranspose)
print(newDataset)
