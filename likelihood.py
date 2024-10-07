import torch
import numpy as np
from scipy.stats import multivariate_normal

latents = torch.randn((12000,14)) # gaussian noise, sample from gaussian mean 0 std 1

# Main sampling loop.
x_next = latents * 165
print(x_next[:,0].mean(), x_next[:,0].std())


data = x_next.cpu().numpy()  # Replace with your actual data; should be an array of 13-dimensional vectors
mu = np.zeros(14)  # Replace with the 13-dimensional mean vector
sigma = np.diag(1 * np.ones(14))  # Replace with the 13x13 covariance matrix

data = x_next.cpu().numpy().reshape(-1)  # Replace with your actual data; should be an array of 13-dimensional vectors
mu = np.zeros(1)  # Replace with the 13-dimensional mean vector
sigma = np.diag(165 * np.ones(1))  # Replace with the 13x13 covariance matrix

# Initialize the multivariate normal distribution
mvn = multivariate_normal(mean=mu, cov=sigma)

# Calculate the probability density of each observation
prob_densities = mvn.pdf(data)

# Calculate the joint likelihood
joint_likelihood = np.sum(prob_densities)

print("Joint Likelihood:", joint_likelihood)


"""
für shapenet: shape 3
"""
mu = 0 * np.ones(3)  # Replace with the 13-dimensional mean vector
sigma = np.diag(165 * np.ones(3))  # Replace with the 13x13 covariance matrix
sample_distribution = np.random.multivariate_normal(mu, sigma, 12000)
# sample_distribution = np.random.uniform(-1000,10000,size=(12000,3))
mu = 0 * np.ones(3)  # Replace with the 13-dimensional mean vector
sigma = np.diag(165 * np.ones(3))  # Replace with the 13x13 covariance matrix
mvn = multivariate_normal(mean=mu, cov=sigma)
prob_densities_sample_dist = mvn.pdf(sample_distribution)
print(prob_densities_sample_dist)
sum_likelihood_sample_dist = np.sum(-np.log(prob_densities_sample_dist+1e-10))
likelihood_sample_dist = sum_likelihood_sample_dist / 12000
print(likelihood_sample_dist)