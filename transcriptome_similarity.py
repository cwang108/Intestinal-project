import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def min_max_normalize(df):
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_normalized

## Read in transcriptome data
transcriptome_file = 'transcriptome.xlsx'
transcriptome = pd.read_excel(transcriptome_file, sheet_name='9groups')

## Define cosine similarity calculation function
def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    magnitude_vector1 = np.linalg.norm(vector1)
    magnitude_vector2 = np.linalg.norm(vector2)
    return dot_product / (magnitude_vector1 * magnitude_vector2)

def calculate_cosine_similarity(transcriptome):
    distances = {}
    for i in range(2, 11):  ## The last four columns are indexed from 2 to 10.
        col_name = transcriptome.columns[i]
        non_zero_data = transcriptome[
            (transcriptome.iloc[:, i] != 0) &
            (transcriptome.iloc[:, 0] != 0) &  ## Control group 0
            (transcriptome.iloc[:, 1] != 0)    ## Control group 1
        ]
        distances[col_name] = [
            cosine_similarity(non_zero_data.iloc[:, i].values,
                              non_zero_data.iloc[:, j].values)
            for j in range(2)  ## The indexes of the first two columns are 0 and 1.
        ]
    return pd.DataFrame(distances, index=['hco11', 'hco13'])

transcriptome_cosine_similarity = calculate_cosine_similarity(transcriptome).T
mean_values = transcriptome_cosine_similarity.iloc[:, :2].mean(axis=1)
transcriptome_cosine_similarity['Mean'] = mean_values
## Output matrix result
print("Transcriptome similarity matrix", transcriptome_cosine_similarity)