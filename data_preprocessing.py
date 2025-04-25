import pandas as pd
import scipy.stats
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

## Percentage of local structural features+local functional features
features_file = 'C:/Users/DELL/Desktop/project/features.xlsx'
features_control = pd.read_excel(features_file, sheet_name='hColon')
features_BL5A5 = pd.read_excel(features_file, sheet_name='BL5A5')
features_BL5A10 = pd.read_excel(features_file, sheet_name='BL5A10')
features_BL5A15 = pd.read_excel(features_file, sheet_name='BL5A15')
features_BL5A20 = pd.read_excel(features_file, sheet_name='BL5A20')
features_BL15 = pd.read_excel(features_file, sheet_name='BL15')
features_NatureWangXia = pd.read_excel(features_file, sheet_name='NatureWangXia')
features_Organoid = pd.read_excel(features_file, sheet_name='Organoid')
features_stomach = pd.read_excel(features_file, sheet_name='stomach')

## Replace NaN values in all data with 0
features_control = features_control.fillna(0)
features_BL5A5 = features_BL5A5.fillna(0)
features_BL5A10 = features_BL5A10.fillna(0)
features_BL5A15 = features_BL5A15.fillna(0)
features_BL5A20 = features_BL5A20.fillna(0)
features_BL15 = features_BL15.fillna(0)
features_NatureWangXia = features_NatureWangXia.fillna(0)
features_Organoid = features_Organoid.fillna(0)
features_stomach = features_stomach.fillna(0)


## Perform max-min normalization on each column.
def min_max_normalize(df):
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_normalized


## Transcriptome data processing
transcriptome_file = 'C:/Users/DELL/Desktop/project/transcriptome.xlsx'
transcriptome = pd.read_excel(transcriptome_file, sheet_name='data')
## Select the desired column.
transcriptome = transcriptome.iloc[:, 1:39]
transcriptome = min_max_normalize(transcriptome)

# Mean substitution repeated experiment
mean_col0 = transcriptome.iloc[:, 0:3].mean(axis=1)
mean_col1 = transcriptome.iloc[:, 3:6].mean(axis=1)
mean_col2 = transcriptome.iloc[:, 6:9].mean(axis=1)
mean_col3 = transcriptome.iloc[:, 9:12].mean(axis=1)
mean_col4 = transcriptome.iloc[:, 12:15].mean(axis=1)
mean_col5 = transcriptome.iloc[:, 15:18].mean(axis=1)
mean_col6 = transcriptome.iloc[:, 18:21].mean(axis=1)
mean_col7 = transcriptome.iloc[:, 21:27].mean(axis=1)
mean_col8 = transcriptome.iloc[:, 27:33].mean(axis=1)
mean_col9 = transcriptome.iloc[:, 33:36].mean(axis=1)
mean_col10 = transcriptome.iloc[:, 36:39].mean(axis=1)

# Combine these mean columns into a new DataFrame
transcriptome = pd.DataFrame({
    'hco11_trans': mean_col0,
    'hco13_trans': mean_col1,
    'BL15_trans': mean_col2,
    'Organoid_trans': mean_col3,
    'BL5A5_trans': mean_col4,
    'BL5A10_trans': mean_col5,
    'BL5A15_trans': mean_col6,
    'BL5A20_trans': mean_col7,
    'NatureWangXia_trans': mean_col8,
    'stomach_trans': mean_col9,
    'liver_trans': mean_col10
})

# Transpose data set
transcriptome = transcriptome.T
transcriptome = pd.DataFrame(transcriptome.T, columns=[
    'hco11_trans', 'hco13_trans', 'BL15_trans', 'Organoid_trans', 'BL5A5_trans', 'BL5A10_trans',
    'BL5A15_trans', 'BL5A20_trans', 'NatureWangXia_trans', 'stomach_trans', 'liver_trans'])