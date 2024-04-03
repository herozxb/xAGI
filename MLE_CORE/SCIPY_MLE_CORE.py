import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

# Generate synthetic data from a normal distribution
np.random.seed(42)
data = np.random.normal(loc=5, scale=2, size=100)

# Define the negative log-likelihood function
def neg_log_likelihood(params, data):
    mu, sigma = params
    log_likelihood = -np.sum(norm.logpdf(data, mu, sigma))
    return log_likelihood

# Initial guess for the parameters
mu_guess = np.mean(data)
sigma_guess = np.std(data)
initial_guess = [mu_guess, sigma_guess]

# Minimize the negative log-likelihood to estimate the parameters
result = minimize(neg_log_likelihood, initial_guess, args=(data,), method='L-BFGS-B', bounds=((None, None), (0.1, None)))
mu_mle, sigma_mle = result.x

# Print the estimated parameters
print("Maximum Likelihood Estimates:")
print("Mean:", mu_mle)
print("Standard Deviation:", sigma_mle)

