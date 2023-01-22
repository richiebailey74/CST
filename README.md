# CST

Class Separation Transformation (CST) is a novel machine learning technique that accomplishes the dual objcetive of significantly reducing the dimensionality of the feature space (down to one dimension) while separating classes as optimally as possible in "clusters" in that single dimension. CST does not preserve any sort of covariance or topologies in the feature space - the only goal is to reduce the dimensionality and so that the mapping provides optimal separation for classifiers. This technique solves a big issue many researchers face in machine learning - the input space reduction and accuracy tradeoff. With our technique, there is no tradeoff between these two properties. The technique is also extremely useful with respect to explainability in ML since the algorithm results in a learned transformation vector that we call f, where each weight corresponds to a particular feature in the original input space. This can lead to a greater understanding in how particular attributes contribute towards defining a features class, which can have enumerable use cases in many fields of research.

Code for Class Separation Transformation experiments are split into directories for each individual contributor.

The Aisharjya directory contains a zip file that can be downloaded containing bash scripts that conduct the paper's experiments and associated data preprocessing.

The Aaditya directory contains follow up experiments with the same data generated from the Aisharjya directory, except done within an ipynb for enhanced readability.

The Richie directory contains experiments conducted with different datasets, the non-hyperparameter tuning experiments, and the associated data preprocessing of the additional data - all in ipynb files. It also has an importable CST.py file containing the CST class.
