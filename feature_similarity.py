import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


features_file = 'feature.xlsx'
features_control = pd.read_excel(features_file, sheet_name='hColon')
features_BL5A5 = pd.read_excel(features_file, sheet_name='BL5A5')
features_BL5A10 = pd.read_excel(features_file, sheet_name='BL5A10')
features_BL5A15 = pd.read_excel(features_file, sheet_name='BL5A15')
features_BL5A20 = pd.read_excel(features_file, sheet_name='BL5A20')
features_BL15 = pd.read_excel(features_file, sheet_name='BL15')
features_NatureWangXia = pd.read_excel(features_file, sheet_name='NatureWangXia')
features_Organoid = pd.read_excel(features_file, sheet_name='Organoid')
features_stomach = pd.read_excel(features_file, sheet_name='stomach')
## Initializing Wasserstein distance matrix
wasserstein_distances = []

for data in [features_BL5A20, features_BL5A15, features_BL5A10, features_BL5A5, features_BL15, features_NatureWangXia,
             features_Organoid, features_stomach]:
    distances = []
    for i in range(features_control.shape[1]):
        control_clean = features_control.iloc[:, i].dropna()
        data_clean = data.iloc[:, i].dropna()
        min_length = min(len(control_clean), len(data_clean))
        control_clean = control_clean.iloc[:min_length]
        data_clean = data_clean.iloc[:min_length]
        ## Calculate Wasserstein distance
        distance = wasserstein_distance(control_clean, data_clean)
        distances.append(distance)
    wasserstein_distances.append(distances)

## Convert the result into a matrix
wasserstein_distances_matrix = pd.DataFrame(wasserstein_distances,
                                            index=['BL5A20', 'BL5A15', 'BL5A10','BL5A5', 'BL15', 'NatureWangXia',
                                                   'Organoid', 'stomach'],
                                            columns=['width', 'height', 'elongation', 'Curvature', 'villiE-angle',
                                                     'cryptE-angle', 'Curl', 'Contour KL', 'Mem-distribution KL',
                                                     'MUC2-distribution KL', 'villiMUC2', 'middleMUC2', 'cryptMUC2',
                                                     'villi/cryptMUC2'])
## Defines the decay rate alpha
'''
Adjust the decay rate alpha according to the sensitivity of each feature to wassertain distance and the difference 
of their importance in measuring similarity. The larger the decay rate alpha, the more sensitive it is.
'''
alpha = [0.01, 0.01, 1.0, 1.0, 0.01, 0.01, 3, 1.0,  1.0,   0.8,   1,   1,   1,   0.2]
## Initializing similarity matrix
similarity_matrix = pd.DataFrame(index=wasserstein_distances_matrix.index, columns=wasserstein_distances_matrix.columns)
## Calculating similarity matrix
for i in range(wasserstein_distances_matrix.shape[1]):
    similarity_matrix.iloc[:, i] = np.exp(-alpha[i] * wasserstein_distances_matrix.iloc[:, i])

## Ensure that the indexes and columns of the similarity matrix are consistent with the original distance matrix
similarity_matrix.index = wasserstein_distances_matrix.index
similarity_matrix.columns = wasserstein_distances_matrix.columns
## Output matrix result
print("Wasserstein distance matrix", wasserstein_distances_matrix)
print("similarity matrix", similarity_matrix)