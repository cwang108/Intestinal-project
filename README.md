# Construction and application of scoring function

## Data preprocessing

Data preprocessing is a critical step in data analysis, aimed at ensuring data quality and applicability to achieve accurate and reliable analytical results. The following code includes missing value processing, normalization, mean substitution, data integration and transposition.&#x20;

Missing data may lead to inaccurate analytical outcomes. By addressing missing values (e.g., replacing them with a placeholder like 0 in this case), the issues of noise and data incompleteness can be mitigated, thereby enhancing the effectiveness of model training. Features often differ in scale and numerical range, and directly comparing or using such features can introduce bias toward certain variables. Max-min normalization (e.g., using MinMaxScaler) scales all data to a uniform range (e.g., 0 to 1), promoting fairness in subsequent analyses and calculations.&#x20;

In repeated experiments, using mean substitution can generate more stable feature expression and reduce random fluctuation in small sample sizes, which is helpful to extract more representative information and improve the reliability of analysis. By integrating multiple columns into a mean value and transposing the data set, we reorganized data in a clearer and more orderly way, which makes the subsequent analysis more concise and efficient. To sum up, data preprocessing significantly improves data quality and usability, laying a solid foundation for further analysis and modeling efforts.

```Python
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
    'BL5A15_trans', 'BL5A20_trans', 'NatureWangXia_trans','stomach_trans', 'liver_trans'])
```

## Calculate the weight matrix

To enable data merging and weight calculation based on the structural and functional characteristics of different experimental groups, a weight matrix is constructed to facilitate the creation of a scoring function. The code implements several key functionalities: data merging, missing values handling, feature separation, weight calculation, and average weight calculation.

&#x20;Firstly, the characteristic data from different experimental groups (such as' BL5A5',' BL5A10',' BL5A15' and' BL5A20') are merged with the corresponding KL values and transcriptome data to form a comprehensive data set. Any missing values in the merged dataset are then addressed to ensure data integrity. Structural features and functional features are defined and separated within the dataset, preparing them for subsequent analysis.

The entropy weight method is employed to calculate the importance weight of each feature. This method assigns weights based on the contribution of each feature to the overall data distribution, thereby reflecting their relative significance in the analysis. For the structural features and functional features of different experimental groups, average weights are calculated, resulting in two new data frames.&#x20;

Finally, by outputting the average weights of structural and functional features (that is, the objective weights in the subjective and objective weighting method), combined with the subjective weights given by researchers according to the importance of each feature, the final structural and functional feature weights are obtained by averaging them, and the construction of the scoring function model is completed.

```Python
## Merged data
combined_BL5A5 = pd.concat([features_BL5A5, KL_BL5A5, transcriptome[['BL5A5_trans']]], axis=1)
combined_BL5A10 = pd.concat([features_BL5A10, KL_BL5A10, transcriptome[['BL5A10_trans']]], axis=1)
combined_BL5A15 = pd.concat([features_BL5A15, KL_BL5A15, transcriptome[['BL5A15_trans']]], axis=1)
combined_BL5A20 = pd.concat([features_BL5A20, KL_BL5A20, transcriptome[['BL5A20_trans']]], axis=1)

## Fill in missing values with fillna
combined_BL5A5.fillna(0, inplace=True)
combined_BL5A10.fillna(0, inplace=True)
combined_BL5A15.fillna(0, inplace=True)
combined_BL5A20.fillna(0, inplace=True)

## Define structural feature column names
structural_features = ['Width', 'Height', 'Elongation', 'Curvature', 'villiE-angle', 'cryptE-angle', 'Curl', 'Contour KL']
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
```

## Calculating feature similarity matrix

Wasserstein distance, also known as Earth Mover's Distance, is a mathematical method to measure the distance between two probability distributions. The definition of Wasserstein distance is based on minimizing the cost of transforming one distribution to the other. While the cost function can vary, it typically represents the effort needed to shift the mass of one distribution to match another. This cost can be expressed as the product of the distance and mass between two locations. The Wasserstein distance is thus the minimum cost across all possible transformation schemes.

Unlike traditional distance metrics, such as Euclidean distance, Wasserstein distance takes into account both the similarity between distributions and the geometric relationships between their components. This makes it particularly well-suited for comparing high-dimensional datasets, where conventional measures might fail to capture nuanced differences.

&#x20;For data other than transcriptome data, we first calculate the Wasserstein distance. That is to say, we regard the data we want to compare with the control group as two distributions, then calculate the difference between the two probability distributions, and then build a similarity matrix based on these distances.

```Python
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
```

## Calculate the similarity of transcriptome data

Cosine similarity is a metric used to measure the similarity between two vectors by calculating the cosine of the angle between them. The cosine value of 0 degree angle is 1, while the cosine value of any other angle is between  -1 and 1. The cosine value of the angle between the two vectors determines whether they point in roughly the same direction. For transcriptome data,which captures global functionality,  we first identify the intersection of genes  shared between two groups (reconstructed intestine and real intestine), and then calculate cosine similarity based on these shared genes to quantify the similarity between the two groups.

```Python
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
```

## Application of scoring function

In the data analysis process outlined above, the weight parameters for the scoring function have been determined. Using these parameters, the structural and functional similarity score matrices of the experimental group, relative to the real human intestine, are weighted element by element to calculate the structural and functional similarity scores, respectively. The overall similarity is then quantified using the F1 score, which provides a comprehensive metric by integrating both structural and functional performance.

The formula is displayed as follows:

```LaTeX
$$  
F_1 = \frac{2 \times similarity_{structure} \times similarity_{functional}}{similarity_{structure} + similarity_{functional}}  
$$
```

## Visualization of scoring results

Since BL15 is a negative control, its structural and functional similarity score is set to 0. To ensure consistency, the scores of other groups in the final results are adjusted by translating them synchronously. After the translation, the shifted target points are scaled back to their initial state (100, 100). This operation is similarly applied to all other groups.  Finally,  the structural and functional similarity scores and F1 scores of the final samples are processed to visualize.

```Python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

results_file = 'C:/Users/DELL/Desktop/project/results.xlsx'
score = pd.read_excel(results_file, sheet_name='Result_all')
structure_score = score['structure']
function_score = score['function']

## Define the range of x and y and create a grid
x = y = np.linspace(-120, 120, 400)
X, Y = np.meshgrid(x, y)
F = np.where(X + Y != 0, (2 * X * Y) / (X + Y), np.nan)

## Set colors and fonts
colors = ['#1D5E22', '#0B7F14', '#64C06C', '#B4DAB7', '#C4C3C1', '#ff7f0e', '#FFEDCB', '#055BA8', '#61A5C2']
plt.rcParams.update({'font.size': 16, 'font.family': 'Arial'})
## Create graphics
fig, ax = plt.subplots(figsize=(8, 6))
labels = ['BL5A20', 'BL5A15', 'BL5A10', 'BL5A5', 'BL15', 'NatureWangXia', 'Organoid', 'Stomach', 'Liver', 'Target']
## N-1 rows of data before drawing
for i in range(len(structure_score) - 1):
    ax.scatter(structure_score[i], function_score[i], color=colors[i], alpha=1, marker='^', s=400, edgecolors='black', label=labels[i], zorder=3)
## Draw the last line of data as a five-pointed star
ax.scatter(structure_score[9], function_score[9], color='red', alpha=1, marker='*', s=500, edgecolors='black', label=labels[9], zorder=3)

## Draw isoline
F_masked = np.where((X < 0) | (Y < 0), np.nan, F)
contour = ax.contour(X, Y, F_masked, levels=[20, 40, 60, 80, 100], linestyles='dashed', colors='#7F7F7F', zorder=1, linewidths=1)

## Set coordinate axes and add graphic borders.
ax.axhline(0, color='black', lw=1.5)
ax.axvline(0, color='black', lw=1.5)
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.set_xlabel('Structure Score', labelpad=100, fontsize=16, color='black')
ax.set_ylabel('Function Score', labelpad=90, fontsize=16, color='black')
ax.legend(bbox_to_anchor=(1.01, 0.5), loc='center left', frameon=False, handleheight=2.5, fontsize=12)
ax.patch.set(edgecolor='black', linewidth='1.5')
## Set the drawing range, scale and scale
ax.set(xlim=(-60, 110), ylim=(-60, 110), aspect='equal', xticks=[-50, 50, 100], yticks=[-50, 0, 50, 100])
ax.tick_params(axis='both', labelsize=13)

## Adjust layout and display graphics
plt.subplots_adjust(right=0.85, left=0.05, bottom=0.15, top=0.9)
## Save the resized picture.
plt.savefig('results/Figs/score.png', dpi=500)
```

For the four groups of BL5A5-BL5A20 with dense score distribution, separate subgraphs are specially made for them to show their similarity score results and differences more clearly.

```Python
## Define the range of x and y and create a grid
x = np.linspace(50, 82, 400)
y = np.linspace(40, 100, 400)
X, Y = np.meshgrid(x, y)
F = np.where(X + Y != 0, (2 * X * Y) / (X + Y), np.nan)
## Set colors and fonts
colors = ['#1D5E22','#0B7F14', '#64C06C', '#B4DAB7']
color_target = ['RED']
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = 'Arial'
## Create graphics
plt.figure(figsize=(6, 5))
labels = ['BL5A5', 'BL5A10', 'BL5A15', 'BL5A20', 'target']
## N-1 rows of data before drawing
for i in range(len(structure_score) - 1):
    plt.scatter(structure_score[i], function_score[i], color=colors[i], alpha=1, marker='^', s=800, edgecolors='black', zorder=3, label=labels[i])
## Draw the last line of data as a five-pointed star
plt.scatter(structure_score[4], function_score[4], color=color_target, alpha=1, marker='*', s=600, edgecolors='black', zorder=3, label=labels[4])

## Draw isoline
levels = [68, 70, 72, 74, 76, 78, 80]
contour = plt.contour(X, Y, F, levels=levels, linestyles='dashed', zorder=1, colors='#7F7F7F')
plt.clabel(contour, inline=True, fmt='%d', fontsize=10, levels=[72, 74, 76, 78, 80])
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
plt.xlim([66, 82])
plt.ylim([66, 82])

## Make sure that the axes are in equal proportions.
plt.gca().set_aspect('equal', adjustable='box')
## Set the drawing range, scale and scale
ax.set(xlim=(66, 82), ylim=(66, 82), aspect='equal', xticks=[70, 75, 80], yticks=[70, 75, 80])
ax.tick_params(axis='both', labelsize=13)
## Set axis label
plt.xlabel('Structure Score', labelpad=10, fontsize=16, color='black')
plt.ylabel('Function Score', labelpad=10, fontsize=16, color='black')
plt.tight_layout()
plt.savefig('results/Figs/score-son.png', dpi=500)
```

