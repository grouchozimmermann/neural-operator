import numpy as np
from numpy.polynomial.chebyshev import Chebyshev


def sampleChebPoly(degree_N, M, num_samples=1):
    """
    Sample a function u(x) as a linear combination of Chebyshev polynomials of the first kind.
    
    Parameters:
    - degree_N (int): Degree of the Chebyshev polynomial (N-1).
    - M (float): Bound for the coefficients, i.e., each coefficient ai is sampled from [-M, M].
    - num_samples (int): Number of polynomial samples to generate.

    Returns:
    - sampled_functions (list): List of sampled functions u(x) represented as callable lambdas.
    - coefficients (list): List of coefficient arrays used for each sampled function.
    """
    sampled_functions = []
    coefficients = []
    
    for _ in range(num_samples):
        # Randomly sample coefficients a_i from [-M, M] for each term up to degree N-1
        a_i = np.random.uniform(-M, M, degree_N)
        coefficients.append(a_i)
        
        # Define the Chebyshev polynomial series u(x) = sum(a_i * T_i(x))
        # Using NumPy's Chebyshev class, we can construct the polynomial directly
        cheb_poly = Chebyshev(a_i)
        
        # Add the sampled polynomial as a function
        sampled_functions.append(cheb_poly)
    
    return sampled_functions, coefficients
