import numpy as np
from Configurations.srgtsPRSCreateGramianMatrix import srgtsPRSCreateGramianMatrix
def srgtsPRSEvaluate(x, srgtSRGT):
    # Simulate srgtSRGT in X
    X = srgtsPRSCreateGramianMatrix(x, srgtSRGT['NbVariables'], srgtSRGT['PRS_Degree'], srgtSRGT['PRS_RemovedIdx'])
    yhat = np.dot(X, srgtSRGT['PRS_Beta'])
    return yhat
