import numpy as np

class CST:
    
    def __init__(self):
        self.f = None
        
    def fit(self, X, y):
        #construct upper triangular A
        sampleNum = X.shape[0] #get the number of samples
        featureNum = X.shape[1] #get the number of features
        temp_A = np.zeros((featureNum, featureNum)) #A has dimensions nxn, devised from outer product of difference of two samples's feature values

        #loops over SxS where S:{1,m} (m = # of samples)
        for i in range(sampleNum-1):
            for j in range(i+1, sampleNum):

                #difference in features between ith and jth samples
                sample_difference = X[i] - X[j]

                #if else block functions as the indicator function
                if y[i] == y[j]:
                    temp_A += (np.outer(sample_difference, sample_difference.T))
                else:
                    temp_A -= (np.outer(sample_difference, sample_difference.T))


        #calculate eigenvalues and eigenvectors of A
        eigvals, eigvecs = np.linalg.eig(temp_A)

        #get minimum eigenvector
        min_ind = np.argmin(eigvals)
        self.f = eigvecs[:,min_ind] #in the algorithm, this is the transformation vector f
        return
        
    def transform(self, X):
        
        if not self.f:
            print("No data has been fit to. Fit to data first!")
            return
        
        #transform the data
        X_trans = X @ self.f

        return X_trans.reshape(-1,1)
    
    def fit_transform(self, X, y):
        #construct upper triangular A
        sampleNum = X.shape[0] #get the number of samples
        featureNum = X.shape[1] #get the number of features
        temp_A = np.zeros((featureNum, featureNum)) #A has dimensions nxn, devised from outer product of difference of two samples's feature values

        #loops over SxS where S:{1,m} (m = # of samples)
        for i in range(sampleNum-1):
            for j in range(i+1, sampleNum):

                #difference in features between ith and jth samples
                sample_difference = X[i] - X[j]

                #if else block functions as the indicator function
                if y[i] == y[j]:
                    temp_A += (np.outer(sample_difference, sample_difference.T))
                else:
                    temp_A -= (np.outer(sample_difference, sample_difference.T))


        #calculate eigenvalues and eigenvectors of A
        eigvals, eigvecs = np.linalg.eig(temp_A)

        #get minimum eigenvector
        min_ind = np.argmin(eigvals)
        self.f = eigvecs[:,min_ind] #in the algorithm, this is the transformation vector f
        
        #transform the data
        X_trans = X @ self.f

        return X_trans.reshape(-1,1)
    
    def transformation_vector(self):
        if not self.f:
            print("No data has been fit to. Fit to data first!")
            return
        
        return self.f
    