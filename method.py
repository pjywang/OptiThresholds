import numpy as np

from scipy.stats import ecdf
from scipy.interpolate import interp1d
from scipy.optimize import LinearConstraint, differential_evolution


class Distribution:
    """
    Class representing the distributional representations wearable device data (e.g., CGM data) and their amalgamated histograms. 
    The class basically stores the distributional data in the form of histograms and quantiles.
    """

    def __init__(self, data, ran=None, M=200):
        """
        Initialize with data and range.

        Parameters:
            data (list): List of wearable device measurements for each subject.
            ran (tuple): Range of data. Default is None, which set minimum and maximum values from the data.
            M (int): Number of quantile grid points for Wasserstein distance computation.
        """
        if ran is not None:
            assert np.min(np.min(data)) >= ran[0] and np.max(np.max(data)) <= ran[1], (
                "Data out of range: specify the correct min/max range of measurement levels"
            )
        else:
            ran = [np.min(np.min(data)) - 1e-8, np.max(np.max(data)) + 1e-8]
        
        self.ran = ran
        self.data = [np.array(datum) for datum in data]
        self.M = M
        self.gr = np.linspace(0, 1, M+2)    # Grid for quantiles
        
        # Get quantile functions and empirical CDFs for each subject
        self.qtiles = self.compute_quantiles()
        self.F_list = self.get_ecdfs()

    def compute_quantiles(self):
        """Compute quantiles for each subject's data."""
        return [np.quantile(data, self.gr) for data in self.data]
    
    def get_ecdfs(self):
        """ Get the empirical CDF callables of the CGM measurements"""
        return [ecdf(data).cdf.evaluate for data in self.data]

    def cutoff_amalgamation(self, cutoffs, fixed=None):
        """
        Compute piecewise-linearized quantiles given the cutoffs, 
        which correspond to the amalgamated histograms based on the cutoffs.

        Parameters:
            cutoffs (list): List of cutoffs (excluding endpoints).
            fixed (list): Optional fixed thresholds for semi-supervised approach.

        Returns:
            list: Amalgamated quantiles.
        """
        if fixed is not None:
            assert np.min(fixed) >= self.ran[0] and np.max(fixed) <= self.ran[1], "Fixed thresholds out of range"
            cutoffs = np.sort(np.r_[fixed, cutoffs])    # Sort is required for the correct interpolation

        q_a_list = list()
        for i in range(len(self.data)):
            knots = np.unique(np.concatenate([[0], self.F_list[i](cutoffs), [1]]))  # endpoints for ecdf 0, 1 values
            q_knots = np.quantile(self.data[i], knots)
            interp_func = interp1d(knots, q_knots, kind='linear', fill_value=(np.min(self.data[i]), np.max(self.data[i])))
            
            # Amalgamated quantiles
            q_a_list.append(interp_func(self.gr)) 
        
        self.q_a = q_a_list
        return q_a_list
    
    def dists_for_Loss1(self, cutoffs, fixed=None, Wdist="W2"):
        """
        Compute Wasserstein distances between original and amalgamated distributions.

        Parameters:
            cutoffs (list): Cutoffs for amalgamation.
            fixed (list): Optional fixed thresholds.
            Wdist (str): Distance metric, "W1" or "W2".

        Returns:
            np.ndarray: List of distances.
        """
        # Compute the quantiles of the amalgamation given cutoffs
        self.cutoff_amalgamation(cutoffs, fixed)

        n = len(self.data)
        distances = np.zeros(n)
        for i in range(n):
            if Wdist == "W2":
                distances[i] = np.sum((self.qtiles[i] - self.q_a[i]) ** 2) / (self.M + 1)
            elif Wdist == "W1":
                distances[i] = np.sum(np.abs(self.qtiles[i] - self.q_a[i])) / (self.M + 1)
        return distances
    
    def Wdist_matrix(self, cutoffs=None, fixed=None, change=True, Wdist="W2"):
        """
        Compute (upper triangular) Wasserstein distance matrix for Loss2 computation.

        Parameters:
            cutoffs (list): Cutoffs for amalgamation.
            fixed (list): Optional fixed thresholds.
            change (bool): Whether to update the amalgamated matrix.
            Wdist (str): Distance metric, "W1" or "W2".

        Returns:
            np.ndarray: Distance matrix.
        """
        n = len(self.data)
        dist_matrix = np.zeros((n, n))
        qtiles = self.qtiles

        if cutoffs is not None:
            self.cutoff_amalgamation(cutoffs, fixed)
            qtiles = self.q_a
        
        for i in range(n-1):
            for j in range(i+1, n):
                if Wdist == "W2":
                    dist = np.sum((qtiles[i] - qtiles[j])**2) / (self.M + 1)
                elif Wdist == "W1":
                    dist = np.sum(np.abs(qtiles[i] - qtiles[j])) / (self.M + 1)
                dist_matrix[i, j] = dist
        
        if cutoffs is None:
            self.dist_matrix = dist_matrix
        elif change:
            self.dist_matrix_amalg = dist_matrix

        return dist_matrix

    def __repr__(self):
        return f"Distributions of {len(self.data)} subjects"


##########################################################
######### Optimizations for data-driven cutoffs ##########
##########################################################

def fitness(cutoffs, data_class, loss, fixed=None, Wdist="W2"):
    """
    Fitness function for the differential evolution algorithm
    """
    if loss == "Loss1":
        return np.mean(data_class.dists_for_Loss1(cutoffs, fixed, Wdist=Wdist))
    elif loss == "Loss2":
        mat_orig = data_class.dist_matrix    # computed only once
        mat_amalg = data_class.Wdist_matrix(cutoffs, fixed=fixed, Wdist=Wdist)
        n = mat_orig.shape[0] 
        return np.sum((mat_orig - mat_amalg) ** 2) * 2 / n / (n - 1)
    

def run_de(data_class, loss, K=4, fixed=None, Wdist="W2",
            maxiter=1000, popsize=15, tol=1e-5, init=None, polish=False,
            disp=False, seed=None, **params):
    """
    Run the differential evolution (DE) algorithm for optimizing cutoffs.

    Parameters:
        data_class (Distribution): data converted to the class Distribution
        loss (string): "Loss1" or "Loss2"
        K (int): Number of cutoffs to optimize.
        fixed (list): Optional fixed thresholds.
        Wdist (str): Distance metric, "W1" or "W2".
        maxiter (int): Maximum number of iterations.
        popsize (int): Population size for the DE algorithm.
        tol (float): Convergence tolerance for DE.
        init (str or None): Initialization method, "custom" or None.
        polish (bool): Whether to refine the result with local optimization.
        disp (bool): Display optimization process.
        seed (int or None): Random seed for reproducibility.
        params (dict): Additional parameters for differential_evolution.

    Returns:
        tuple: Optimized cutoffs and the minimum loss value.
    """
    assert loss in ["Loss1", "Loss2", "BC"], "Invalid loss function"

    # Precompute distance matrix for Loss2
    if loss == "Loss2":
        data_class.Wdist_matrix(Wdist=Wdist)

    ran = data_class.ran
    bounds = [ran] * K  # Bounds for cutoffs

    # Linear constraint for monotonicity
    constr_matrix = (np.eye(K, k=1) - np.eye(K))[:-1]
    linear_constraint = LinearConstraint(constr_matrix, 
                                         lb=0., ub=np.inf) if K > 1 else ()
    
    result = differential_evolution(
        func=fitness,
        bounds=bounds,
        args=(data_class, loss, fixed, Wdist),
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,       # Relative tolerance for convergence,
        seed=seed,
        callback=None,
        disp=disp,
        polish=polish,
        constraints=linear_constraint,
        **params
    )
    
    res = np.sort(np.r_[fixed, result.x]) if fixed is not None else result.x
    best_cutoffs = np.concatenate(([ran[0]], res, [ran[1]]))
    min_loss = result.fun
    
    return best_cutoffs, min_loss



### Greedy approaches for threshold optimization

def agglomerative_discrete(data_class, K, loss, thresholds=None, Wdist="W2", verbose=False):
    """
    Iteratively merge thresholds to achieve the target number of bins using a greedy approach.

    Parameters:
        data_class (Distribution): Instance of Distribution class.
        K (int): Target number of thresholds.
        loss (str): Loss function type, "Loss1" or "Loss2".
        thresholds (list): Initial thresholds; if None, defaults to range-based thresholds.
        Wdist (str): Distance metric, "W1" or "W2".
        verbose (bool): Whether to print progress.

    Returns:
        dict: Final thresholds and history of changes.
    """
    assert loss in ["Loss1", "Loss2"], "Invalid loss function specified"

    if thresholds is None:
        thresholds = list(np.arange(data_class.ran[0] + 1, data_class.ran[1] + 1, 1))

    if loss == "Loss2":
        data_class.Wdist_matrix(Wdist=Wdist)  # precompute the distance matrix for the original data

    loss_history = {}
    thres_history = [thresholds]

    while len(thresholds) > K:
        best_loss = np.inf
        best_threshold = None

        # Evaluate optimal threshold to delete
        for potential_threshold in thresholds:
            # Remove the potential threshold and calculate loss
            temp_thresholds = thresholds.copy()
            temp_thresholds.remove(potential_threshold)
            loss_val = fitness(temp_thresholds, data_class, loss=loss, Wdist=Wdist)

            if loss_val < best_loss:
                best_loss = loss_val
                best_threshold = potential_threshold

        # Remove the threshold minimizing the loss
        thresholds.remove(best_threshold)
        if verbose:
            print("Removed threshold:", best_threshold)

        loss_history[len(thresholds)] = best_loss
        thres_history.append([thresholds.copy()])

    print("Final loss:", best_loss)
    
    return {
        'thresholds': thresholds,
        'loss_history': loss_history,
        'thresholds_history': thres_history
    }


def divisive_discrete(data_class, K, loss, thresholds=None, thre_list=None, Wdist="W2"):
    """
    Stepwise divisive thresholding of histograms based on given loss function.

    Parameters:
        data_class (Distribution): Instance of Distribution class.
        K (int): Target number of thresholds.
        loss (str): Loss function type, "Loss1" or "Loss2".
        thresholds (list): Initial fixed thresholds.
        thre_list (list): List of candidate thresholds; defaults to integer values in the data range.
        Wdist (str): Distance metric, "W1" or "W2".

    Returns:
        dict: Final thresholds and their history.
    """
    assert loss in ["Loss1", "Loss2"], "Invalid loss function string"

    loss_history = {}
    thres_history = []

    if thresholds is None:
        thresholds = []
    else:
        assert isinstance(thresholds, list), "Fixed thresholds must be a list"

    if thre_list is None:
        thre_list = np.arange(data_class.ran[0] + 1, data_class.ran[1] + 1, 1)
    
    if not all(threshold in thre_list for threshold in thresholds):
        raise ValueError("Invalid fixed thresholds. Some values are not in thre_list.")

    if loss == "Loss2":
        # precompute the distance matrix for the original data
        data_class.Wdist_matrix(Wdist=Wdist)   

    # Iterative search for the optimal thresholds
    while len(thresholds) < K:
        best_loss = np.inf
        best_threshold = None

        # Evaluate potential new thresholds
        for potential_threshold in thre_list:
            if potential_threshold in thresholds:
                continue

            # Insert the potential threshold and calculate loss
            temp_thresholds = sorted(thresholds + [potential_threshold])
            loss_val = fitness(temp_thresholds, data_class, loss=loss, Wdist=Wdist)

            if loss_val < best_loss:
                best_loss = loss_val
                best_threshold = potential_threshold

        # Update thresholds with the best found threshold
        thresholds.append(best_threshold)
        thresholds.sort()
        print("Picked thresholds:", thresholds)

        loss_history[len(thresholds)] = best_loss
        thres_history.append(best_threshold)

    print("Final loss:", best_loss)

    return {
        'thresholds': thresholds,
        'loss_history': loss_history,
        'thresholds_history': thres_history
    }


###########################################################
# Bray-Curtis criterion for agglomerative amalgamation (the method PAA)
# Adapted from the paper: Principal amalgamation analysis for microbiome data (2022), by Li et al.

def agglomerative_BC(data_class, K, thresholds=None, verbose=False):
    """
    Agglomerative method for reducing thresholds based on Bray-Curtis criterion.

    Parameters:
        data_class (Distribution): Instance of Distribution class.
        K (int): Target number of thresholds.
        thresholds (list): Initial thresholds; if None, defaults integer values in the data range
        verbose (bool): Whether to print progress.

    Returns:
        dict: Final thresholds and their history.
    """
    if thresholds is None:
        thresholds = list(np.arange(data_class.ran[0] - 0.5, data_class.ran[1] + 0.5, 1))
    a, b = data_class.ran[0] - 0.5, data_class.ran[1] + 0.5
    cutoffs = np.r_[a, thresholds, b]

    n_obs = np.sum(np.histogram(data_class.data[0], bins=cutoffs)[0])
    compo_list = [np.histogram(data_class.data[i], bins=cutoffs)[0] / n_obs for i in range(len(data_class.data))]
    compo_matrix = np.array(compo_list)

    thres_history = []

    while len(thresholds) > K:
        criterion = np.zeros(len(thresholds))
        for l in range(len(thresholds)):
            criterion[l] = BC_criterion(compo_matrix, l)

        l = np.argmin(criterion)
        removed = thresholds.pop(l)
        if verbose:
            print("Removed threshold:", removed)
        compo_matrix[:, l] += compo_matrix[:, l+1]
        compo_matrix = np.delete(compo_matrix, l+1, axis=1)
        thres_history.append(thresholds.copy())

    return {
        "thresholds": thresholds,
        "thres_history": thres_history
    }


def BC_criterion(compo_matrix, l):
    """
    Compute Bray-Curtis criterion for merging thresholds, derived by Li et al. (2022).

    Parameters:
        compo_matrix (np.ndarray): Composition matrix.
        l (int): Index of threshold to evaluate.

    Returns:
        float: Computed Bray-Curtis criterion value.
    """
    n = compo_matrix.shape[0]
    val = 0
    for i in range(n):
        for j in range(i + 1, n):
            val += (min(compo_matrix[i, l], compo_matrix[j, l]) + min(compo_matrix[i, l+1], compo_matrix[j, l+1]) - 
                    min(compo_matrix[i, l] + compo_matrix[i, l+1], compo_matrix[j, l] + compo_matrix[j, l+1])) ** 2
    return val



# def bray_curtis(cutoffs, data_class, fixed=None):
#     if fixed is not None:
#         assert np.min(fixed) >= data_class.ran[0] and np.max(fixed) <= data_class.ran[1], "Fixed thresholds out of range"
#         # Sort is required for the correct interpolation
#         cutoffs = np.sort(np.r_[fixed, cutoffs])

#     n_obs = np.sum(np.histogram(data_class.data[0], bins=cutoffs)[0])
#     compo_list = [np.histogram(data_class.data[i], bins=cutoffs)[0] / n_obs for i in range(len(data_class.data))]

#     # Compute the Bray-Curtis distance
#     dist = 0
#     for i in range(len(data_class.data)):
#         for j in range(i+1, len(data_class.data)):
#             dist += np.sum(np.abs(compo_list[i] - compo_list[j])) / 2

#     return dist



##############################################################
##############################################################
##############################################################
# Deprecated

# In run_de function, initialization
    # # Custom Latin hypercube initialization (monotonically covers the range sliced by the quantiles)
    # if init == "custom":
    #     qtile_list = np.quantile(data_class.data, np.linspace(0, 1, K + 1))
    #     np.random.seed(seed)
    #     inits = np.random.uniform(size=popsize)[:, None] * np.diff(qtile_list) + qtile_list[0:K]
    # else:
    #     inits = 'latinhypercube'


# In fitness,
    # elif loss == "BC":
    #     return bray_curtis(cutoffs, data_class, fixed=fixed)


# def quantile_hist(x, ran=(40, 400), M=200):
#     """
#     Compute the empirical quantiles of CGM measurements x based on the histogram model, compatible with ecdf_histogram
#     Returns a callable quantile function

#     Well, this is not needed (overcomplication) since we can directly compute the quantiles from the empirical CDF
#     """
#     hist, bins = np.histogram(x, bins=np.arange(ran[0] - 0.5, ran[1] + 1.5, 1), density=True)
#     cdf = np.concatenate(([0.], np.cumsum(hist)))

#     # First indices of the unique cdf values except the first zero, using jump locations detected by np.diff
#     first_unique_indices = np.unique(np.diff(cdf).nonzero()[0]) + 1
#     unique_cdf = cdf[first_unique_indices]
#     unique_q_values = bins[first_unique_indices]


#     return


# Class methods for the Distribution class
    # def ecdf_hist(self, cutoffs=None, fixed=None):
    #     """
    #     Compute the empirical CDF of CGM measurements x or their amalgamated versions based on the cutoffs
    #     By default, histogram-density model is applied for the empirical CDFs, setting the midpoints of integral values as the breakpoints
        
    #     - Function for stepwise amalgamation of histograms & W1 distance approaches
        
    #     param cutoffs: list of cutoffs for the amalgamation, EXCLUDING endpoints.
    #                     If None, the midpoints of integral values are used
        
    #     Returns a list of callable CDF functions
    #     """
    #     if cutoffs is None:
    #         cutoffs = np.arange(self.ran[0] + 0.5, self.ran[1] + 0.5, 1)

    #     if fixed is not None:
    #         assert np.min(fixed) >= self.ran[0] and np.max(fixed) <= self.ran[1], "Fixed thresholds out of range"
    #         # Sort is required for the correct interpolation
    #         cutoffs = np.sort(np.r_[fixed, cutoffs])

    #     breaks = np.r_[self.ran[0] - 0.5, cutoffs, self.ran[1] + 0.5]

    #     fun_list = list()
    #     for data in self.data:
    #         hist, bins = np.histogram(data, bins=breaks, density=True)
    #         cdf = np.concatenate(([0.], np.cumsum(hist)))
    #         fun_list.append(interp1d(bins, cdf, kind='linear', fill_value=(0., 1.), bounds_error=False))
    #     return fun_list


    # def W1dist_for_Loss1(self, cutoffs, fixed=None):
    #     """
    #     Compute the total Wasserstein-1 distance for the cutoffs
    #     """
    #     # Compute the quantiles of the amalgamation given cutoffs
    #     F_list = self.ecdf_hist()
    #     F_amalg_list = self.ecdf_hist(cutoffs, fixed)

    #     gr = np.arange(self.ran[0] - 0.5, self.ran[1] + 1.5, 1)
    #     n = len(self.data)
    #     distances = np.zeros(n)
    #     for i in range(n):
    #         # Integration of |F(x) - F_amalg(x)| dx
    #         distances[i] = trapezoid(np.abs(F_list[i](gr) - F_amalg_list[i](gr)), x=gr)
    #     return distances


    # def W1dist_matrix(self, cutoffs=None, fixed=None, change=True):
    #     """
    #     Wasserstein-1 distance matrix of the original data or amalgamated data by cutoffs
    #     computed directly from the empirical CDFs
    #     """
    #     n = len(self.data)
    #     dist_matrix = np.zeros((n, n))

    #     # Empirical cdfs; piecewise linear interpolated based on the cutoffs
    #     F_list = self.ecdf_hist(cutoffs, fixed)

    #     gr = np.arange(self.ran[0] - 0.5, self.ran[1] + 1.5, 1)

    #     for i in range(n-1):
    #         for j in range(i+1, n):
    #             # integration of |F_i(x) - F_j(x)| dx
    #             dist = trapezoid(np.abs(F_list[i](gr) - F_list[j](gr)), x=gr)
    #             # np.sum(np.abs(F_list[i](gr) - F_list[j](gr))) / n_points
    #             dist_matrix[i, j] = dist
    #             # dist_matrix[j, i] = dist
        
    #     if cutoffs is None:
    #         self.dist_matrix = dist_matrix
    #     elif change:
    #         self.dist_matrix_amalg = dist_matrix

    #     return dist_matrix


# def gradient_approach(data_class, K, loss, fixed=None, Wdist="W2", tol=1e-6, disp=True, maxiter=1000):
#     """
#     Run Monotonicity constraints are imposed.

#     Parameters
#     ----------
#     data_class: class Distribution, data converted to the class Distribution
#     K: int, the target number of thresholds
#     loss: string, "Loss1" or "Loss2"
#     fixed: list of fixed thresholds for semi-supervised approach
#     Wdist: string, "W1" or "W2", Wasserstein -1 or -2 distance
#     """
#     assert loss in ["Loss1", "Loss2"], "Invalid loss function"

#     if loss == "Loss2":
#         data_class.Wdist_matrix(Wdist=Wdist) # precompute the distance matrix for the original data

#     # range of the data, typically [40, 400]
#     ran = data_class.ran
#     bounds = [ran] * K  # Bounds for cutoffs
    
#     # Define the linear constraint for monotonicity
#     constr_matrix = (np.eye(K, k=1) - np.eye(K))[:-1]
#     linear_constraint = LinearConstraint(constr_matrix, 
#                                          lb=0., ub=np.inf)
    
#     # Equi-distance like initialization
#     if fixed is not None:
#         assert np.min(fixed) >= ran[0] and np.max(fixed) <= ran[1], "Fixed thresholds out of range"
#         # Sort is required for the correct interpolation
#         init = np.sort(np.r_[fixed, np.linspace(ran[0] - 0.5, ran[1] + 0.5, K + 2 - len(fixed))[1:-1]])
#     else:
#         init = np.linspace(ran[0] - 0.5, ran[1] + 0.5, K + 2)[1:-1]

#     result = minimize(
#         fun=fitness,
#         x0=init,
#         args=(data_class, loss, fixed, Wdist),
#         method='SLSQP',
#         jac=None,
#         hess=None,
#         hessp=None,
#         bounds=bounds,
#         constraints=linear_constraint,
#         tol=tol,
#         callback=None,
#         options={'maxiter': maxiter, 'disp': disp}
#     )
    
#     res = np.sort(np.r_[fixed, result.x]) if fixed is not None else result.x

#     best_cutoffs = np.concatenate(([ran[0]], res, [ran[1]]))
#     min_loss = result.fun
    
#     return best_cutoffs, min_loss



# from itertools import combinations, islice
# from math import comb

# def exhaustive_search(data_class, K, loss, Wdist="W2", njobs=1, batch_size=int(1e5)):
#     """
#     Exhaustive search for the optimal thresholds based on the specified loss function.

#     Parameters
#     ----------
#     data_class : Distribution
#         Data converted to the class Distribution.
#     K : int
#         The target number of thresholds.
#     loss : str
#         "Loss1" or "Loss2".
#     Wdist : str, optional
#         "W1" or "W2", Wasserstein -1 or -2 distance. Default is "W2".
#     njobs : int, optional
#         Number of parallel jobs. Default is 1.
#     batch_size : int, optional
#         Number of combinations to process per batch. Adjust based on memory constraints. Default is 10,000.

#     Returns
#     -------
#     dict
#         Contains 'thresholds' (best thresholds found) and 'loss' (corresponding loss value).
#     """
    
#     # Validate the loss function input
#     assert loss in ["Loss1", "Loss2"], "Invalid loss function string"
    
#     if loss == "Loss2":
#         data_class.Wdist_matrix(Wdist=Wdist) # precompute the distance matrix for the original data

#     # Initialize the best loss and corresponding thresholds
#     best_loss = np.inf
#     best_thresholds = None
    
#     # Generate all possible threshold values based on the data range
#     threshold_values = np.arange(data_class.ran[0] - 0.5, data_class.ran[1] + 1.5, 1)
    
#     # Total number of combinations
#     total_combinations = comb(len(threshold_values), K)
#     print(f"Total combinations to evaluate: {total_combinations}")
    
#     # Generator to yield combinations in batches
#     def batched_combinations(iterable, n):
#         """Yield successive n-sized batches from iterable."""
#         it = iter(iterable)
#         while True:
#             batch = list(islice(it, n))
#             if not batch:
#                 break
#             yield batch
    
#     # Function to compute loss for a single combination
#     def compute_loss(thresholds):
#         return fitness(thresholds, data_class, loss=loss, Wdist=Wdist), thresholds
    
#     with Parallel(n_jobs=njobs) as parallel:
#         # Iterate over combinations in batches
#         for batch_num, batch in enumerate(batched_combinations(combinations(threshold_values, K), batch_size), 1):
#             print(f"Processing batch {batch_num} out of {total_combinations // batch_size + 1}...")
            
#             # Compute losses in parallel for the current batch
#             batch_results = parallel(
#                 delayed(compute_loss)(thresholds) for thresholds in batch
#             )
            
#             # Find the best in the current batch
#             batch_best_loss, batch_best_thresholds = min(batch_results, key=lambda x: x[0])
            
#             # Update global best if necessary
#             if batch_best_loss < best_loss:
#                 best_loss = batch_best_loss
#                 best_thresholds = batch_best_thresholds
            
#             # print(f"Batch {batch_num} best loss: {batch_best_loss}")
    
#     # Final result
#     print(f"\nFinal best loss: {best_loss}")
#     print(f"Best thresholds: {best_thresholds}")
    
#     return {
#         'thresholds': best_thresholds,
#         'loss': best_loss
#     }


# def amalgam_Loss1(cdf, t):
#     """
#     Calculate the loss 1 function for amalgamating the bin at t at each sample.
#     Adjacent bins separated by t are amalgamated and the loss 1 is calculated.

#     Note: This function does NOT delete the row t since we need to search optimal t (delete later).
    
#     Parameters
#         :param cdf: pd.DataFrame, representing the piecewise-linear cdf (thresholds, cdf_vals)
#         :param t: int, the threshold value to be amalgamated and deleted

#     Return: float, the loss value
#     """

#     thlds = np.array(cdf['thresholds'])
#     # If t is not in the intermediate row names, warning and end
#     if t not in thlds[1:-1]:
#         print("The threshold value is not in the row names of the cumulative histogram")
#         return None
    
#     t_pos = np.where(thlds == t)[0][0]
#     cdf_vals = np.array(cdf['cdf_vals'])
    
#     # Endpoints of the bins needed for computation
#     a1, a2, a3 = cdf_vals[t_pos - 1], cdf_vals[t_pos], cdf_vals[t_pos + 1]
#     b1, b2, b3 = thlds[t_pos - 1], t, thlds[t_pos + 1]

#     if a1 == a3:
#         return 0
#     else:
#         # L2 distance between before/after quantile-linearization (amalgamation)
#         integral = 0
        
#         # Left part (a1 to a2)
#         if a1 != a2:
#             alpha = (b3 - b1) / (a3 - a1)
#             beta = (b2 - b1) / (a2 - a1)
#             integral += (alpha - beta)**2 * (a2 - a1)**3 / 3
        
#         # Right part (a2 to a3)
#         if a2 != a3:
#             alpha = (b3 - b1) / (a3 - a1)
#             gamma = (b3 - b2) / (a3 - a2)
#             integral += (alpha - gamma)**2 * (a3 - a2)**3 / 3
    
#     return integral


# def agglomerative_Loss1(data_class, K, cdf_list=None, njobs=1):
#     """
#     Stepwise agglomerative amalgamation of the thresholds of the histograms based on the Loss 1 function (assuming integer observations like CGM).
#     The Loss 1 function is calculated for each threshold and the threshold with the smallest Loss 1 is selected, followed by amalgamation at that threshold.

#     Parameters
#         :param data_class: class Distribution, data converted to the class Distribution
#         :param K: int, the target number of thresholds
#         :param cdf_list: list of pd.DataFrame, cumulative histograms with cutoffs and cdf cdf values. 
#                          If not None, further stepwise amalgamation is performed based on the given cdf_list.
#         :param njobs: int, the number of parallel jobs for the loss 1 calculation subprocess (very small impact; njobs=4 seems optimal)
#     """

#     if cdf_list is None:
#         # make data to cumulative histograms
#         cutoffs = np.arange(data_class.ran[0] - 0.5, data_class.ran[1] + 1.5, 1)
#         cdfs = data_class.ecdf_hist()

#         cdf_list = []
#         for i in range(len(data_class.data)):
#             cdf_list.append(pd.DataFrame(data={'thresholds': cutoffs, 'cdf_vals': cdfs[i](cutoffs)},
#                                          index=cutoffs)) # set the row names as the thresholds


#     n = len(cdf_list)
#     num_thlds = len(cdf_list[0]) - 2  # intermediate thresholds
#     history = []  # history of remaining thresholds
#     loss_history = []  # loss history before selection
#     Loss1_history = []  # Selected optimal Loss1 history

#     for L in range(num_thlds, K, -1):
#         # Remaining thresholds during the amalgamation process (excluding endpoints)
#         row_names = cdf_list[0]['thresholds'].iloc[1:-1]
#         history.append(np.array(row_names))

#         loss1 = []
#         to_amalgam = None
#         print("Amalgamating from", L, "thresholds", end=": ")
        
#         def loss1_func(j):
#             t_j = row_names.iloc[j]
#             return sum(amalgam_Loss1(cdf, t_j) for cdf in cdf_list)
        
#         jobs = min(njobs, L)
#         loss1 = Parallel(n_jobs=jobs)(delayed(loss1_func)(j) for j in range(L))
#         loss_history.append(loss1)

#         # The threshold with the smallest loss1 is selected
#         if to_amalgam is None:
#             to_amalgam = row_names.iloc[np.argmin(loss1)]
#         # print("Selected:", to_amalgam)

#         # Amalgamation == "deletion" of the row to_amalgam
#         print(" deleting threshold", to_amalgam)
#         # print(cdf_list[0])
#         for i in range(n):
#             cdf_list[i] = cdf_list[i].drop(to_amalgam)

#         # Save the actual Loss 1 function history "after amalgamation"
#         cutoffs = [data_class.ran[0] - 0.5] + list(cdf_list[0]['thresholds'].iloc[1:-1]) + [data_class.ran[1] + 0.5]
#         Loss1_history.append(sum(data_class.W2dist_for_Loss1(cutoffs)))

#     return {
#         'amalgamated_data': cdf_list,
#         'thresholds': np.array(cdf_list[0]['thresholds'])[1:-1],
#         'history': history + [np.array(cdf_list[0]['thresholds'])[1:-1]],
#         'loss_history': loss_history,
#         'Loss1_history': Loss1_history
#     }
