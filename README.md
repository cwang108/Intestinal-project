# Construction and application of scoring function

## Data preprocessing

Data preprocessing is a critical step in data analysis, aimed at ensuring data quality and applicability to achieve accurate and reliable analytical results. The following code includes missing value processing, normalization, mean substitution, data integration and transposition.&#x20;

Missing data may lead to inaccurate analytical outcomes. By addressing missing values (e.g., replacing them with a placeholder like 0 in this case), the issues of noise and data incompleteness can be mitigated, thereby enhancing the effectiveness of model training. Features often differ in scale and numerical range, and directly comparing or using such features can introduce bias toward certain variables. Max-min normalization (e.g., using MinMaxScaler) scales all data to a uniform range (e.g., 0 to 1), promoting fairness in subsequent analyses and calculations.&#x20;

In repeated experiments, using mean substitution can generate more stable feature expression and reduce random fluctuation in small sample sizes, which is helpful to extract more representative information and improve the reliability of analysis. By integrating multiple columns into a mean value and transposing the data set, we reorganized data in a clearer and more orderly way, which makes the subsequent analysis more concise and efficient. To sum up, data preprocessing significantly improves data quality and usability, laying a solid foundation for further analysis and modeling efforts.

## Calculate the weight matrix

To enable data merging and weight calculation based on the structural and functional characteristics of different experimental groups, a weight matrix is constructed to facilitate the creation of a scoring function. The code implements several key functionalities: data merging, missing values handling, feature separation, weight calculation, and average weight calculation.

&#x20;Firstly, the characteristic data from different experimental groups (such as' BL5A5',' BL5A10',' BL5A15' and' BL5A20') are merged with the corresponding KL values and transcriptome data to form a comprehensive data set. Any missing values in the merged dataset are then addressed to ensure data integrity. Structural features and functional features are defined and separated within the dataset, preparing them for subsequent analysis.

The entropy weight method is employed to calculate the importance weight of each feature. This method assigns weights based on the contribution of each feature to the overall data distribution, thereby reflecting their relative significance in the analysis. For the structural features and functional features of different experimental groups, average weights are calculated, resulting in two new data frames.&#x20;

Finally, by outputting the average weights of structural and functional features (that is, the objective weights in the subjective and objective weighting method), combined with the subjective weights given by researchers according to the importance of each feature, the final structural and functional feature weights are obtained by averaging them, and the construction of the scoring function model is completed.

## Calculating feature similarity matrix

Wasserstein distance, also known as Earth Mover's Distance, is a mathematical method to measure the distance between two probability distributions. The definition of Wasserstein distance is based on minimizing the cost of transforming one distribution to the other. While the cost function can vary, it typically represents the effort needed to shift the mass of one distribution to match another. This cost can be expressed as the product of the distance and mass between two locations. The Wasserstein distance is thus the minimum cost across all possible transformation schemes.

Unlike traditional distance metrics, such as Euclidean distance, Wasserstein distance takes into account both the similarity between distributions and the geometric relationships between their components. This makes it particularly well-suited for comparing high-dimensional datasets, where conventional measures might fail to capture nuanced differences.

&#x20;For data other than transcriptome data, we first calculate the Wasserstein distance. That is to say, we regard the data we want to compare with the control group as two distributions, then calculate the difference between the two probability distributions, and then build a similarity matrix based on these distances.

## Calculate the similarity of transcriptome data

Cosine similarity is a metric used to measure the similarity between two vectors by calculating the cosine of the angle between them. The cosine value of 0 degree angle is 1, while the cosine value of any other angle is between  -1 and 1. The cosine value of the angle between the two vectors determines whether they point in roughly the same direction. For transcriptome data,which captures global functionality,  we first identify the intersection of genes  shared between two groups (reconstructed intestine and real intestine), and then calculate cosine similarity based on these shared genes to quantify the similarity between the two groups.

## Application of scoring function

In the data analysis process outlined above, the weight parameters for the scoring function have been determined. Using these parameters, the structural and functional similarity score matrices of the experimental group, relative to the real human intestine, are weighted element by element to calculate the structural and functional similarity scores, respectively. The overall similarity is then quantified using the F1 score, which provides a comprehensive metric by integrating both structural and functional performance.

## Visualization of scoring results

Since BL15 is a negative control, its structural and functional similarity score is set to 0. To ensure consistency, the scores of other groups in the final results are adjusted by translating them synchronously. After the translation, the shifted target points are scaled back to their initial state (100, 100). This operation is similarly applied to all other groups.  Finally,  the structural and functional similarity scores and F1 scores of the final samples are processed to visualize.

