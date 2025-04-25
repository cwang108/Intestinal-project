import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from transcriptome_similarity import transcriptome

features_file = 'features.xlsx'
features_control = pd.read_excel(features_file, sheet_name='hColon')
features_BL5A5 = pd.read_excel(features_file, sheet_name='BL5A5')
features_BL5A10 = pd.read_excel(features_file, sheet_name='BL5A10')
features_BL5A15 = pd.read_excel(features_file, sheet_name='BL5A15')
features_BL5A20 = pd.read_excel(features_file, sheet_name='BL5A20')
## Merged data
combined_BL5A5 = pd.concat([features_BL5A5, transcriptome[['BL5A5_trans']]], axis=1)
combined_BL5A10 = pd.concat([features_BL5A10, transcriptome[['BL5A10_trans']]], axis=1)
combined_BL5A15 = pd.concat([features_BL5A15, transcriptome[['BL5A15_trans']]], axis=1)
combined_BL5A20 = pd.concat([features_BL5A20, transcriptome[['BL5A20_trans']]], axis=1)
## Fill in missing values with fillna
combined_BL5A5.fillna(0, inplace=True)
combined_BL5A10.fillna(0, inplace=True)
combined_BL5A15.fillna(0, inplace=True)
combined_BL5A20.fillna(0, inplace=True)
## Define structural feature column names
structural_features = ['Width', 'Height', 'Elongation', 'Curvature', 'villiE-angle', 'cryptE-angle', 'Curl',
                       'Contour KL']
functional_features = ['villiMUC2', 'middleMUC2', 'cryptMUC2', 'villi/cryptMUC2', 'Mem-distribution KL',
                       'MUC2-distribution KL', 'BL5A5_trans']
## Split data set
def split_data(data):
    structural = data[structural_features]
    functional = data.drop(columns=structural_features)
    return structural, functional
## Apply split function
structural_BL5A5, functional_BL5A5 = split_data(combined_BL5A5)
structural_BL5A10, functional_BL5A10 = split_data(combined_BL5A10)
structural_BL5A15, functional_BL5A15 = split_data(combined_BL5A15)
structural_BL5A20, functional_BL5A20 = split_data(combined_BL5A20)
## Entropy weight method to calculate weight
def mylog(p):
    n = len(p)
    lnp = np.zeros(n)
    for i in range(n):
        if p[i] == 0:
            lnp[i] = 0
        else:
            lnp[i] = np.log(p[i])
    return lnp


def calculate_weights(data):
    n, m = data.shape
    D = np.zeros(m)
    for i in range(m):
        x = data.iloc[:, i]
        p = x / np.sum(x)
        e = -np.sum(p * mylog(p)) / np.log(n)
        D[i] = 1 - e
    W = D / np.sum(D)
    return W


## Structure and function are separated to calculate the weight of each data set.
W_structural_BL5A5 = calculate_weights(structural_BL5A5)
W_functional_BL5A5 = calculate_weights(functional_BL5A5)
W_structural_BL5A10 = calculate_weights(structural_BL5A10)
W_functional_BL5A10 = calculate_weights(functional_BL5A10)
W_structural_BL5A15 = calculate_weights(structural_BL5A15)
W_functional_BL5A15 = calculate_weights(functional_BL5A15)
W_structural_BL5A20 = calculate_weights(structural_BL5A20)
W_functional_BL5A20 = calculate_weights(functional_BL5A20)
## Calculate the average weight of structural features and functional features.
structural_weights = np.array([
    W_structural_BL5A5,
    W_structural_BL5A10,
    W_structural_BL5A15,
    W_structural_BL5A20
])
functional_weights = np.array([
    W_functional_BL5A5,
    W_functional_BL5A10,
    W_functional_BL5A15,
    W_functional_BL5A20
])

## Calculate the average weight
avg_structural_weights = np.mean(structural_weights, axis=0)
avg_functional_weights = np.mean(functional_weights, axis=0)
## Create a DataFrame to store the average weight.
avg_structural_weights = pd.DataFrame([avg_structural_weights], columns=structural_features, index=['weight'])
avg_functional_weights = pd.DataFrame([avg_functional_weights], columns=functional_features, index=['weight'])
## Structural and functional weight results
print(avg_structural_weights)
print(avg_functional_weights)