import numpy as np
from scipy.linalg import pinv
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from Configurations.srgtsPRSCreateGramianMatrix import srgtsPRSCreateGramianMatrix

def srgtsPRSFit(srgtOPT):
    # Initialize srgtSRGT dictionary
    srgtSRGT = {}

    # Get basic information
    srgtSRGT['NbPoints'], srgtSRGT['NbVariables'] = srgtOPT['P'].shape
    srgtSRGT['PRS_Degree'] = srgtOPT['PRS_Degree']

    # Generate Gramian matrix X
    X = srgtsPRSCreateGramianMatrix(srgtOPT['P'], srgtSRGT['NbVariables'], srgtOPT['PRS_Degree'])

    # Compute beta coefficients and statistics
    if srgtOPT['PRS_Regression'] == 'Full':
        model = LinearRegression(fit_intercept=False)
        model.fit(X, srgtOPT['T'])
        beta = model.coef_
        residuals = srgtOPT['T'] - model.predict(X)
        df = srgtSRGT['NbPoints'] - srgtSRGT['NbVariables']
        mse = np.sum(residuals ** 2) / df
        SE = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * mse)
        PRS_RemovedIdx = []

    elif srgtOPT['PRS_Regression'] == 'StepwiseSRGTS':
        y = srgtOPT['T']
        Xinv = pinv(X.T @ X)
        beta = Xinv @ X.T @ y
        NbCoeff = len(beta)
        t_statistic = beta / np.sqrt(
            ((y.T @ y - (beta.T @ X.T @ y)) / (srgtSRGT['NbPoints'] - NbCoeff)) * np.diag(Xinv))
        idx = np.argmin(np.abs(t_statistic))
        PRS_RemovedIdx = []
        ctr = 1
        while np.abs(t_statistic[idx]) < 1:
            PRS_RemovedIdx.append(idx)
            PRS_RemovedIdx.sort()
            ctr += 1
            X = np.delete(X, idx, axis=1)
            Xinv = pinv(X.T @ X)
            beta = Xinv @ X.T @ y
            NbCoeff = len(beta)
            t_statistic = beta / np.sqrt(
                ((y.T @ y - (beta.T @ X.T @ y)) / (srgtSRGT['NbPoints'] - NbCoeff)) * np.diag(Xinv))
            idx = np.argmin(np.abs(t_statistic))
        residuals = y - X @ beta
        SE = np.sqrt(np.sum(residuals ** 2) / (srgtSRGT['NbPoints'] - NbCoeff))

    elif srgtOPT['PRS_Regression'] == 'StepwiseMATLAB':
        model = LinearRegression(fit_intercept=False)
        model.fit(X, srgtOPT['T'])
        beta = model.coef_
        _, pvals = f_regression(X, srgtOPT['T'])
        PRS_RemovedIdx = np.where(pvals > 0.05)[0]
        beta = beta[PRS_RemovedIdx]
        residuals = srgtOPT['T'] - model.predict(X)
        SE = np.sqrt(np.mean(residuals ** 2))

    elif srgtOPT['PRS_Regression'] == 'ZeroIntercept':
        X = np.delete(X, 0, axis=1)
        model = LinearRegression(fit_intercept=False)
        model.fit(X, srgtOPT['T'])
        beta = model.coef_
        residuals = srgtOPT['T'] - model.predict(X)
        df = srgtSRGT['NbPoints'] - srgtSRGT['NbVariables']
        mse = np.sum(residuals ** 2) / df
        SE = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * mse)
        PRS_RemovedIdx = [0]

    # Assign srgtSRGT outputs
    srgtSRGT['PRS_Beta'] = beta
    srgtSRGT['PRS_SE'] = SE
    srgtSRGT['PRS_RemovedIdx'] = PRS_RemovedIdx

    return srgtSRGT

# You would need to define the srgtsPRSCreateGramianMatrix function separately.

# Example usage:
# srgtOPT = {
#     'P': X,
#     'T': Y,
#     'PRS_Degree': 2,
#     'PRS_Regression': 'Full'
# }
# surrogate = srgtsPRSFit(srgtOPT)


