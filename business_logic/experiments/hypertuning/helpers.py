# standard
import pandas as pd
import numpy as np

# classifiers
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# reduction techniques
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import KernelPCA
from business_logic.algorithmic_implementation.CST_implementation import CST

# preprocessing and transformation
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# analysis
from sklearn.metrics import balanced_accuracy_score

def experiments_BACs_nonCST(X_train, X_test, y_train, y_test, k):

    pca = PCA(n_components = k)
    pca.fit(X_train)
    X_train_trans = pca.transform(X_train)
    X_test_trans = pca.transform(X_test)
    pca_bacs = experiment_BACs_helper(X_train_trans, X_test_trans, y_train, y_test, k)

    selectKBest = SelectKBest(score_func=mutual_info_classif, k=k)
    selectKBest.fit(X_train, y_train)
    X_train_trans = selectKBest.transform(X_train)
    X_test_trans = selectKBest.transform(X_test)
    mi_bacs = experiment_BACs_helper(X_train_trans, X_test_trans, y_train, y_test, k)

    mapper = umap.UMAP(n_components=k).fit(X_train, y_train) # may need to fix this
    X_train_trans = mapper.transform(X_train)
    X_test_trans = mapper.transform(X_test)
    umap_bacs = experiment_BACs_helper(X_train_trans, X_test_trans, y_train, y_test, k)

    kpca = KernelPCA(n_components = k, kernel = 'poly')
    kpca.fit(X_train)
    X_train_trans = kpca.transform(X_train)
    X_test_trans = kpca.transform(X_test)
    kpca_bacs = experiment_BACs_helper(X_train_trans, X_test_trans, y_train, y_test, k)

    return  pca_bacs, mi_bacs, umap_bacs, kpca_bacs


def experiment_BACs_helper(X_train_trans, X_test_trans, y_train, y_test, k):
    temp_BACs = np.zeros(6)

    neighbors = KNeighborsClassifier(n_neighbors=3)
    neighbors.fit(X_train_trans, y_train)
    temp_BACs[0] = balanced_accuracy_score(y_test, neighbors.predict(X_test_trans))

    svm = SVC(kernel='linear', C=1)
    svm.fit(X_train_trans, y_train)
    temp_BACs[1] = balanced_accuracy_score(y_test, svm.predict(X_test_trans))

    gnb = GaussianNB()
    gnb.fit(X_train_trans, y_train)
    temp_BACs[2] = balanced_accuracy_score(y_test, gnb.predict(X_test_trans))

    rf = RandomForestClassifier(random_state=137)
    rf.fit(X_train_trans, y_train)
    temp_BACs[3] = balanced_accuracy_score(y_test, rf.predict(X_test_trans))

    sgd = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
    sgd.fit(X_train_trans, y_train)
    temp_BACs[4] = balanced_accuracy_score(y_test, sgd.predict(X_test_trans))

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_trans, y_train)
    temp_BACs[5] = balanced_accuracy_score(y_test, lda.predict(X_test_trans))

    return temp_BACs


# checks out bc 50 total iterations from for loops and each iteration conducts 30 ML experiments
def runexperiments_full_2(g1_samples, g2_samples, k):
    x = pd.concat([g1_samples, g2_samples])  # add together all postiive and negative labels
    x = StandardScaler().fit_transform(x)  # standardize values in x (z-score standardization)

    target_g1 = pd.DataFrame(np.zeros((len(g1_samples), 1)))  # get number of target group 1
    target_g2 = pd.DataFrame(np.ones((len(g2_samples), 1)))  # get number of target group 2

    target = pd.concat([target_g1, target_g2])  # concatenate target zeros ones twos etc together
    target = target.reset_index(drop=True)

    splits_rats = np.array([.1, .15, .2, .25, .3])

    BAC_sums_pca = np.zeros(6)
    BAC_sums_mi = np.zeros(6)
    BAC_sums_umap = np.zeros(6)
    BAC_sums_kpca = np.zeros(6)

    split_num = 5

    for i in splits_rats:
        # split the data into test and train
        X_train, X_test, y_train, y_test = train_test_split(x, target.to_numpy(), test_size=i, random_state=42)
        y_train = y_train.ravel()
        y_test = y_test.ravel()

        kf = KFold(n_splits=split_num, shuffle=True, random_state=42)

        for trainIn, valIn in kf.split(X_train):
            x_train_nonval = X_train[trainIn]
            y_train_nonval = y_train[trainIn]

            x_val = X_train[valIn]
            y_val = y_train[valIn]

            x_temp = np.concatenate((x_val, X_test), axis=0)
            y_temp = np.concatenate((y_val, y_test), axis=0)

            t2, t3, t4, t5 = experiments_BACs_nonCST(x_train_nonval, x_temp, y_train_nonval, y_temp, k)
            BAC_sums_pca += t2
            BAC_sums_mi += t3
            BAC_sums_umap += t4
            BAC_sums_kpca += t5

    numIter = splits_rats.shape[0] * split_num
    BAC_sums_pca /= numIter
    BAC_sums_mi /= numIter
    BAC_sums_umap /= numIter
    BAC_sums_kpca /= numIter

    return BAC_sums_pca, BAC_sums_mi, BAC_sums_umap, BAC_sums_kpca

def getAllNonCSTValues(pos,neg):
    dictTracker = {}
    for i in range(2,203,10):
        exp_pca, exp_mi, exp_umap, exp_kpca = runexperiments_full_2(pos, neg, i)
        dictTracker[i] = [exp_pca, exp_mi, exp_umap, exp_kpca]
    return dictTracker


def experiments_BACs_CST(X_train, X_test, y_train, y_test, k):
    # CST goes here
    cst = CST()
    X_train_trans, f = cst.fit(X_train, y_train)
    X_test_trans = cst.transform(X_test)
    cst_bacs = experiment_BACs_helper(X_train_trans, X_test_trans, y_train, y_test, 2)

    return cst_bacs


# checks out bc 50 total iterations from for loops and each iteration conducts 30 ML experiments
def runexperiments_full_2_CST(g1_samples, g2_samples):
    x = pd.concat([g1_samples, g2_samples])  # add together all postiive and negative labels
    x = StandardScaler().fit_transform(x)  # standardize values in x (z-score standardization)

    target_g1 = pd.DataFrame(np.zeros((len(g1_samples), 1)))  # get number of target group 1
    target_g2 = pd.DataFrame(np.ones((len(g2_samples), 1)))  # get number of target group 2

    target = pd.concat([target_g1, target_g2])  # concatenate target zeros ones twos etc together
    target = target.reset_index(drop=True)

    splits_rats = np.array([.1, .15, .2, .25, .3])

    BAC_sums_cst = np.zeros(6)

    for i in splits_rats:
        # split the data into test and train
        X_train, X_test, y_train, y_test = train_test_split(x, target.to_numpy(), test_size=i, random_state=42)
        y_train = y_train.ravel()
        y_test = y_test.ravel()

        t1 = experiments_BACs_CST(X_train, X_test, y_train, y_test, 2)
        BAC_sums_cst += t1

    numIter = splits_rats.shape[0]
    BAC_sums_cst /= numIter

    return BAC_sums_cst
