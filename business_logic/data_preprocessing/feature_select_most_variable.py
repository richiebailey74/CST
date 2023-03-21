import numpy as np


# method to select the n most variable features for the particular experiment
def write_n_most_variable(experimentName, pos_samples, neg_samples, n):
    featureSize = pos_samples.shape[1]
    relativeDifferences = np.zeros(featureSize)

    for i in range(0 ,featureSize):

        relativeDifferences[i] = np.abs(pos_samples.iloc[: ,i].mean() -
                                        neg_samples.iloc[: ,i].mean()) / np.minimum(pos_samples.iloc[: ,i].mean(),
                                                                                   neg_samples.iloc[: ,i].mean())
        if relativeDifferences[i] == np.inf:
            relativeDifferences[i] = np.abs(pos_samples.iloc[: ,i].mean() -
                                            neg_samples.iloc[: ,i].mean()) / np.maximum(pos_samples.iloc[: ,i].mean(),
                                                                                       neg_samples.iloc[: ,i].mean())
        if np.isnan(relativeDifferences[i]):
            relativeDifferences[i] = 0.0

    n_most_variable_indices = np.argsort(relativeDifferences)[::-1][:n]
    ind = np.sort(n_most_variable_indices)
    df_to_write_pos = pos_samples.iloc[: ,ind]
    df_to_write_neg = neg_samples.iloc[: ,ind]
    df_to_write_pos.to_csv("data/" + experimentName + "_pos_samples.tsv", sep='\t')
    df_to_write_neg.to_csv("data/" + experimentName + "_neg_samples.tsv", sep='\t')
    return