# A Python file of functions generating various simulation data

import numpy as np
from scipy.stats import truncnorm


def simul_uniform_mixture(n_samples, setting="Setting1", eps = 0, seed=None, **params):
    """
    Simulate mixture of uniform distributions data with given weights

    Parameters
    ----------
    n_samples : int
        number of distribution samples
    setting: str
        "Setting1" or "Setting2" for the different settings of the weights (see the paper)
    eps : float
        Standard deviation of the Gaussian noise added to the thresholds
    seed : int, default=None
        Random seed for the random number generator
    **params : additional parameters to pass to the function `generate_mixture`
        In case one wants to use different n_obs or thresholds in the simulation

    Returns
    -------
    samples : np.ndarray of shape (n_samples, n_obs)
        Simulated data for each distribution sample
    """
    params.setdefault('n_obs', 1000)

    # Different seeds for different noise levels/settings
    if seed is not None:
        seed += eps * 1000
        if setting == "Setting2":
            seed += 50000

    RS = np.random.RandomState(seed)

    # Weights generation based on the Setting 1/2
    if setting == "Setting1":
        alpha = (.3, .4, .2, .1)
        weight_matrix = RS.dirichlet(np.array(alpha) * 20, n_samples)

    elif setting == "Setting2":
        fixed_weights=(.2, .1)
        w1, w2 = RS.dirichlet(np.array([.5, .5]) * 5, n_samples).T * (1 - np.sum(fixed_weights))
        
        # stack the copys of fixed weights through the rows
        fixed_weights_tile = np.tile(fixed_weights, (n_samples, 1))   
        weight_matrix =  np.column_stack((w1, w2, fixed_weights_tile))

    else:
        raise ValueError("Invalid setting: " + setting)

    # Generate empirical mixture distributions
    samples = np.zeros((n_samples, params['n_obs']))
    for i in range(n_samples):
        samples[i] = generate_mixture(weight_matrix[i], eps=eps, random_state=RS, **params)
    
    return samples


def generate_mixture(weights, thresholds=(40, 70, 180, 250, 400), n_obs=1000, eps=0, random_state=None):
    """
    Generate observations from a mixture of uniform distributions given weights vector

    Parameters
    ----------
    weights : np.ndarray 
        Nonnegative weights of the mixture components, summing to 1
    thresholds : array-like indexable vector. 
        Uniform distributions are drawn from the intervals (thresholds[i], thresholds[i+1])
        Must satisfy thresholds[0] < thresholds[1] < ... < thresholds[-1], and len(weights) == len(thresholds) - 1
    n_obs : int, default 1000
        Number of observations to generate
    eps : float
        Standard deviation of the Gaussian noise added to the thresholds
    random_state : np.random.RandomState, default=None 
        Random number generator, often inherited from the parent function
    """
    n_intervals = len(weights)
    assert n_intervals == len(thresholds) - 1
    assert np.all(np.diff(thresholds) > 0)

    RS = random_state if random_state is not None else np.random.RandomState(None)
    
    thresholds = np.array(thresholds, dtype=float)
    
    # Add noise to the intermediate thresholds
    if eps > 0:
        # Truncated normal distribution to [-30, 30]
        a, b = -30 / eps, 30 / eps
        noise = truncnorm.rvs(a, b, scale=eps, size=len(thresholds) - 2, random_state=RS)
        thresholds[1:-1] += noise

    # Generate mixture uniform samples
    samples = np.zeros(n_obs)
    for i in range(n_obs):
        # Choose an index from the np.arange(n_intervals) with mixture weights
        idx = RS.choice(n_intervals, p=weights)

        # Draw a uniform sample
        samples[i] = RS.uniform(thresholds[idx], thresholds[idx + 1])

    return samples


###################################
### Deprecated functions

# def generate_mixture(weights, thresholds=(40, 100, 200, 400), n_obs=1000, var=0., eps=0., seed=None):
#     """
#     Generate data from a mixture of uniform distributions given weights vector

#     Parameters
#     ----------
#     weights : np.ndarray 
#         Nonnegative weights of the mixture components, summing to 1
#     thresholds : array-like indexable vector. 
#         Uniform distributions are drawn from the intervals (thresholds[i], thresholds[i+1])
#         Must satisfy thresholds[0] < thresholds[1] < ... < thresholds[-1], and len(weights) == len(thresholds) - 1
#     n : int, default=2000
#         Number of observations to generate
#     var : float 
#         Standard deviation of the Gaussian noise added to the samples
#     eps : float
#         Standard deviation of the Gaussian noise added to the thresholds
#     seed : int, default=None
#     """
#     n_intervals = len(weights)
#     RS = np.random.RandomState(seed)
#     assert n_intervals == len(thresholds) - 1
#     assert np.all(np.diff(thresholds) > 0)
#     assert var >= 0.
    
#     # Add noise to the intermediate thresholds
#     thresholds = np.array(thresholds, dtype=float)
#     thresholds[1:-1] += RS.normal(0, eps, len(thresholds) - 2)

#     # Clip the thresholds to the range [40, 400] to avoid thresholds going out of bounds
#     thresholds = np.clip(thresholds, 40, 400)

#     # Generate samples from the uniform distributions
#     samples = np.zeros(n_obs)
#     for i in range(n_obs):
#         # Choose an index from the np.arange(n_intervals) with mixture weights
#         idx = RS.choice(n_intervals, p=weights)

#         # Draw a sample from the chosen component
#         samples[i] = RS.uniform(thresholds[idx], thresholds[idx + 1])
        
#     # Add Gaussian noise to the generated samples of the uniform mixture
#     samples += np.random.normal(0, np.sqrt(var), n_obs)

#     return samples


# def weight_draws(n_samples, alpha=(.5, .3, .2), c=20., seed=None):
#     """
#     Generate weights for the mixture components using a Dirichlet distribution
#     with the parameter alpha * c

#     The parameter c controls the concentration of the distribution
#     """
#     RS = np.random.RandomState(seed)
#     return RS.dirichlet(np.array(alpha) * c, n_samples)


# def simul_uniform_mixture(n_samples, weight_matrix=None, seed=None, **params):
#     """
#     Simulate mixture of uniform distributions data with given weights

#     Parameters
#     n_samples : int
#         number of distribution samples
#     weight_matrix : np.ndarray of shape (n_samples, n_intervals), default=None
#         Nonnegative weights of the mixture components
#         Use `weight_draws` to generate the weights
#     **params : additional parameters to pass to the function `generate_mixture`
#         Using this, we can specify the fixed thresholds.
#     """
#     params.setdefault('n_obs', 1000)

#     if weight_matrix is None:
#         weight_matrix = weight_draws(n_samples, seed=seed)                            

#     samples = np.zeros((n_samples, params['n_obs']))
#     for i in range(n_samples):
#         seed_i = seed + i if seed is not None else None
#         samples[i] = generate_mixture(weight_matrix[i], seed=seed_i, **params)
    
#     return samples


# def weight2(n_samples, fixed_weights=(.2, .1), c=5., seed=None):
#     """
#     Generate weights for the mixture components w_1, w_2, fixed_weights, so w_1 + w_2 = 1 - sum(fixed_weights) 
#     using a Dirichlet distribution for w_1, w_2 with the parameter c * (0.5, 0.5), followed by scaling by 1 - sum(fixed_weights)
#     """
#     RS = np.random.RandomState(seed)
#     w1, w2 = RS.dirichlet(np.array([.5, .5]) * c, n_samples).T * (1 - np.sum(fixed_weights))
#     # stack the copys of fixed weights through the rows
#     fixed_weights = np.tile(fixed_weights, (n_samples, 1))
#     return np.column_stack((w1, w2, fixed_weights))