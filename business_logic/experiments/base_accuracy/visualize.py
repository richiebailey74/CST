import numpy as np
import matplotlib.pyplot as plt


# method that takes in the arrays of BAC values to display
def graph_BAC_results(cstBACs, pcaBACs, miBACs, umapBACs, kpcaBACs, experiment):

    x = np.arange(6)
    cstBACs = np.ndarray.tolist(cstBACs)
    pcaBACs = np.ndarray.tolist(pcaBACs)
    miBACs = np.ndarray.tolist(miBACs)
    umapBACs = np.ndarray.tolist(umapBACs)
    kpcaBACs = np.ndarray.tolist(kpcaBACs)

    width = 0.86 / 6

    colors = ['#000000', '#404040', '#7f7f7f', '#bfbfbf', '#ffffff']
    DR = ['CST', 'PCA', 'MI', 'UMAP', 'kPCA']
    classifiers = ["kNN", "SVM", "GNB", "RF", "SGD", "LDA"]

    plt.figure(figsize=(15, 10))
    plt.bar(x - 2 * width, cstBACs, width, color='#000000', edgecolor='black')
    plt.bar(x - width, pcaBACs, width, color='#404040', edgecolor='black')
    plt.bar(x, miBACs, width, color='#7f7f7f', edgecolor='black')
    plt.bar(x + width, umapBACs, width, color='#bfbfbf', edgecolor='black')
    plt.bar(x + 2 * width, kpcaBACs, width, color='#ffffff', edgecolor='black')

    plt.xticks(x, classifiers)
    plt.xlabel("DR / Feature Selection")
    plt.ylabel("Balanced Accuracy scores")
    plt.legend(DR, fontsize=8)
    plt.title(experiment)
    plt.show()
    return
