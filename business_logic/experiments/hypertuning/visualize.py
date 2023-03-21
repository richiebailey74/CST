import matplotlib.pyplot as plt

def generate_plots(all_vals, separate=True, together=True):

    if not separate and not together:
        raise Exception("Both arguments cannot be false because no graphs are to be shown!")

    # generate lists for each classifier
    # order of the DR techniques in the dict are: PCA, MI, UMAP, KPCA, CST
    # order of the classifiers in the arrays are: KNN, SVM, GNB, RF, SGD, LDA
    feature_ticks = list(range(2, 203, 10))
    PCA_KNN = []
    PCA_SVM = []
    PCA_GNB = []
    PCA_RF = []
    PCA_SGD = []
    PCA_LDA = []

    MI_KNN = []
    MI_SVM = []
    MI_GNB = []
    MI_RF = []
    MI_SGD = []
    MI_LDA = []

    UMAP_KNN = []
    UMAP_SVM = []
    UMAP_GNB = []
    UMAP_RF = []
    UMAP_SGD = []
    UMAP_LDA = []

    KPCA_KNN = []
    KPCA_SVM = []
    KPCA_GNB = []
    KPCA_RF = []
    KPCA_SGD = []
    KPCA_LDA = []

    CST_KNN = []
    CST_SVM = []
    CST_GNB = []
    CST_RF = []
    CST_SGD = []
    CST_LDA = []

    for i in range(2, 203, 10):
        PCA_KNN.append(all_vals[i][0][0])
        PCA_SVM.append(all_vals[i][0][1])
        PCA_GNB.append(all_vals[i][0][2])
        PCA_RF.append(all_vals[i][0][3])
        PCA_SGD.append(all_vals[i][0][4])
        PCA_LDA.append(all_vals[i][0][5])

        MI_KNN.append(all_vals[i][1][0])
        MI_SVM.append(all_vals[i][1][1])
        MI_GNB.append(all_vals[i][1][2])
        MI_RF.append(all_vals[i][1][3])
        MI_SGD.append(all_vals[i][1][4])
        MI_LDA.append(all_vals[i][1][5])

        UMAP_KNN.append(all_vals[i][2][0])
        UMAP_SVM.append(all_vals[i][2][1])
        UMAP_GNB.append(all_vals[i][2][2])
        UMAP_RF.append(all_vals[i][2][3])
        UMAP_SGD.append(all_vals[i][2][4])
        UMAP_LDA.append(all_vals[i][2][5])

        KPCA_KNN.append(all_vals[i][3][0])
        KPCA_SVM.append(all_vals[i][3][1])
        KPCA_GNB.append(all_vals[i][3][2])
        KPCA_RF.append(all_vals[i][3][3])
        KPCA_SGD.append(all_vals[i][3][4])
        KPCA_LDA.append(all_vals[i][3][5])

        CST_KNN.append(all_vals[i][4][0])
        CST_SVM.append(all_vals[i][4][1])
        CST_GNB.append(all_vals[i][4][2])
        CST_RF.append(all_vals[i][4][3])
        CST_SGD.append(all_vals[i][4][4])
        CST_LDA.append(all_vals[i][4][5])

    if separate:
        plt.title("KNN Classifier for All Reduction Techniques")
        plt.xlabel("Numer of Features")
        plt.ylabel("Balanced Accuracy")
        plt.plot(feature_ticks, CST_KNN, label="CST")
        plt.plot(feature_ticks, KPCA_KNN, label="KPCA")
        plt.plot(feature_ticks, PCA_KNN, label="PCA")
        plt.plot(feature_ticks, MI_KNN, label="MI")
        plt.plot(feature_ticks, UMAP_KNN, label="UMAP")
        plt.legend()
        plt.savefig('figures/hypertune_knn.png')
        plt.show()

        plt.title("SVM Classifier for All Reduction Techniques")
        plt.xlabel("Numer of Features")
        plt.ylabel("Balanced Accuracy")
        plt.plot(feature_ticks, CST_SVM, label="CST")
        plt.plot(feature_ticks, KPCA_SVM, label="KPCA")
        plt.plot(feature_ticks, PCA_SVM, label="PCA")
        plt.plot(feature_ticks, MI_SVM, label="MI")
        plt.plot(feature_ticks, UMAP_SVM, label="UMAP")
        plt.legend()
        plt.savefig('figures/hypertune_svm.png')
        plt.show()

        plt.title("GNB Classifier for All Reduction Techniques")
        plt.xlabel("Numer of Features")
        plt.ylabel("Balanced Accuracy")
        plt.plot(feature_ticks, CST_GNB, label="CST")
        plt.plot(feature_ticks, KPCA_GNB, label="KPCA")
        plt.plot(feature_ticks, PCA_GNB, label="PCA")
        plt.plot(feature_ticks, MI_GNB, label="MI")
        plt.plot(feature_ticks, UMAP_GNB, label="UMAP")
        plt.legend()
        plt.savefig('figures/hypertune_gnb.png')
        plt.show()

        plt.title("RF Classifier for All Reduction Techniques")
        plt.xlabel("Numer of Features")
        plt.ylabel("Balanced Accuracy")
        plt.plot(feature_ticks, CST_RF, label="CST")
        plt.plot(feature_ticks, KPCA_RF, label="KPCA")
        plt.plot(feature_ticks, PCA_RF, label="PCA")
        plt.plot(feature_ticks, MI_RF, label="MI")
        plt.plot(feature_ticks, UMAP_RF, label="UMAP")
        plt.legend()
        plt.savefig('figures/hypertune_rf.png')
        plt.show()

        plt.title("SGD Classifier for All Reduction Techniques")
        plt.xlabel("Numer of Features")
        plt.ylabel("Balanced Accuracy")
        plt.plot(feature_ticks, CST_SGD, label="CST")
        plt.plot(feature_ticks, KPCA_SGD, label="KPCA")
        plt.plot(feature_ticks, PCA_SGD, label="PCA")
        plt.plot(feature_ticks, MI_SGD, label="MI")
        plt.plot(feature_ticks, UMAP_SGD, label="UMAP")
        plt.legend()
        plt.savefig('figures/hypertune_sgd.png')
        plt.show()

        plt.title("LDA Classifier for All Reduction Techniques")
        plt.xlabel("Numer of Features")
        plt.ylabel("Balanced Accuracy")
        plt.plot(feature_ticks, CST_LDA, label="CST")
        plt.plot(feature_ticks, KPCA_LDA, label="KPCA")
        plt.plot(feature_ticks, PCA_LDA, label="PCA")
        plt.plot(feature_ticks, MI_LDA, label="MI")
        plt.plot(feature_ticks, UMAP_LDA, label="UMAP")
        plt.legend()
        plt.savefig('figures/hypertune_lda.png')
        plt.show()

    if together:
        plt.subplot(3, 2, 1)
        plt.title("KNN Classifier for All Reduction Techniques")
        plt.xlabel("Numer of Features")
        plt.ylabel("Balanced Accuracy")
        plt.plot(feature_ticks, CST_KNN, label="CST")
        plt.plot(feature_ticks, KPCA_KNN, label="KPCA")
        plt.plot(feature_ticks, PCA_KNN, label="PCA")
        plt.plot(feature_ticks, MI_KNN, label="MI")
        plt.plot(feature_ticks, UMAP_KNN, label="UMAP")
        plt.legend()

        plt.subplot(3, 2, 2)
        plt.title("GNB Classifier for All Reduction Techniques")
        plt.xlabel("Numer of Features")
        plt.ylabel("Balanced Accuracy")
        plt.plot(feature_ticks, CST_GNB, label="CST")
        plt.plot(feature_ticks, KPCA_GNB, label="KPCA")
        plt.plot(feature_ticks, PCA_GNB, label="PCA")
        plt.plot(feature_ticks, MI_GNB, label="MI")
        plt.plot(feature_ticks, UMAP_GNB, label="UMAP")
        plt.legend()

        plt.subplot(3, 2, 3)
        plt.title("SVM Classifier for All Reduction Techniques")
        plt.xlabel("Numer of Features")
        plt.ylabel("Balanced Accuracy")
        plt.plot(feature_ticks, CST_SVM, label="CST")
        plt.plot(feature_ticks, KPCA_SVM, label="KPCA")
        plt.plot(feature_ticks, PCA_SVM, label="PCA")
        plt.plot(feature_ticks, MI_SVM, label="MI")
        plt.plot(feature_ticks, UMAP_SVM, label="UMAP")
        plt.legend()

        plt.subplot(3, 2, 4)
        plt.title("RF Classifier for All Reduction Techniques")
        plt.xlabel("Numer of Features")
        plt.ylabel("Balanced Accuracy")
        plt.plot(feature_ticks, CST_RF, label="CST")
        plt.plot(feature_ticks, KPCA_RF, label="KPCA")
        plt.plot(feature_ticks, PCA_RF, label="PCA")
        plt.plot(feature_ticks, MI_RF, label="MI")
        plt.plot(feature_ticks, UMAP_RF, label="UMAP")
        plt.legend()

        plt.subplot(3, 2, 5)
        plt.title("SGD Classifier for All Reduction Techniques")
        plt.xlabel("Numer of Features")
        plt.ylabel("Balanced Accuracy")
        plt.plot(feature_ticks, CST_SGD, label="CST")
        plt.plot(feature_ticks, KPCA_SGD, label="KPCA")
        plt.plot(feature_ticks, PCA_SGD, label="PCA")
        plt.plot(feature_ticks, MI_SGD, label="MI")
        plt.plot(feature_ticks, UMAP_SGD, label="UMAP")
        plt.legend()

        plt.subplot(3, 2, 6)
        plt.title("LDA Classifier for All Reduction Techniques")
        plt.xlabel("Numer of Features")
        plt.ylabel("Balanced Accuracy")
        plt.plot(feature_ticks, CST_LDA, label="CST")
        plt.plot(feature_ticks, KPCA_LDA, label="KPCA")
        plt.plot(feature_ticks, PCA_LDA, label="PCA")
        plt.plot(feature_ticks, MI_LDA, label="MI")
        plt.plot(feature_ticks, UMAP_LDA, label="UMAP")
        plt.legend()

        plt.show()
    return
