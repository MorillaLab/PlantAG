"""Analytic Pearson residuals (verbatim from the PlantAG notebook, cell 23)."""
import numpy as np




def pearson_residuals(counts, theta, clipping=True):
    '''Computes analytical residuals for NB model with a fixed theta, clipping outlier residuals to sqrt(N)'''
    counts_sum0 = np.sum(counts, axis=0)
    counts_sum1 = np.sum(counts, axis=1)
    counts_sum  = np.sum(counts)

    #get residuals
    mu = counts_sum1 @ counts_sum0 / counts_sum
    z = (counts - mu) / np.sqrt(mu + mu/theta + 0.01)

    #clip to sqrt(n)
    if clipping:
        n = counts.shape[0]
        z[z >  np.sqrt(n)] =  np.sqrt(n)
        z[z < -np.sqrt(n)] = -np.sqrt(n)

    return z
