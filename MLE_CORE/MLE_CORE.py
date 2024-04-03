import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data from a normal distribution
np.random.seed(42)
data = np.random.normal(loc=5, scale=2, size=100)

# Define the negative log-likelihood function
def neg_log_likelihood(params, data):
    mu, sigma = params
    log_likelihood = -np.sum(-0.5 * np.log(2 * np.pi * sigma ** 2) - 0.5 * ((data - mu) / sigma) ** 2)
    return -log_likelihood

# Initial guess for the parameters
mu_guess = np.mean(data)
sigma_guess = np.std(data)
initial_guess = [mu_guess, sigma_guess]

# Minimize the negative log-likelihood to estimate the parameters using grid search
def estimate_parameters(data):
    best_params = None
    best_likelihood = -np.inf
    for mu in np.linspace(np.min(data), np.max(data), 100):
        for sigma in np.linspace(0.1, np.max(data) - np.min(data), 100):
            likelihood = neg_log_likelihood([mu, sigma], data)
            if likelihood > best_likelihood:
                best_likelihood = likelihood
                best_params = [mu, sigma]
    return best_params

# Estimate parameters using MLE
mu_mle, sigma_mle = estimate_parameters(data)

# Print the estimated parameters
print("Maximum Likelihood Estimates 2:")
print("Mean:", mu_mle)
print("Standard Deviation:", sigma_mle)



# Estimated parameters
mu_mle = mu_mle
sigma_mle = sigma_mle

# Generate points for the fitted Gaussian distribution
x = np.linspace(np.min(data), np.max(data), 100)
y = 1 / (sigma_mle * np.sqrt(2 * np.pi)) * np.exp(-(x - mu_mle)**2 / (2 * sigma_mle**2))

# Plot the original data and fitted Gaussian distribution
plt.figure(figsize=(8, 6))
plt.hist(data, bins=20, density=True, alpha=0.5, color='blue', label='Original Data')
plt.plot(x, y, color='red', label='Fitted Gaussian Distribution')
plt.title('Original Data and Fitted Gaussian Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()
