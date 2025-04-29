import os
import numpy as np
from scipy.integrate import solve_ivp
from sampleChebyPoly import sampleChebPoly

def generateDataFromChebFunc1D(sampledFunc,nPoints):
    """
    Creates and saves two files with input and output data for use in training the DeepONet architecture. 
    The input data will be generated from Chebyshev Polynomial functions, the output data will be the antiderivate of said data.
    Antiderivatives will be solved by RK45

    Parameters: 
    - sampledFunc (list): List of np Chebyshev functions
    - nPoints (int): Number of discretized points
    """

    x = np.linspace(0,1,nPoints)
    n = len(sampledFunc)

    inData = np.empty([n, nPoints])
    outData = np.empty([n, nPoints])

    for i,func in enumerate(sampledFunc):
        func = sampledFunc[i]
        terre = lambda x,u: func(x)
        inData[i,:] = func(x)
        outData[i,:] = solve_ivp(terre, [0,1], [0],t_eval=x).y
    
    # Define the target directory and filename
    target_directory = './src/data/antiderivative'

    # filename for the file you want to save
    inDataFileName = f"ChebRHS{num_samples}"
    outDataFileName = f"ChebAntiDerivative{num_samples}"
    
    # Ensure the directory exists (optional)
    os.makedirs(target_directory, exist_ok=True)

    np.save(os.path.join(target_directory, inDataFileName), inData)
    np.save(os.path.join(target_directory, outDataFileName), outData)
    
# Example usage:
degree_N = 4  # For example, degree N=5 uses T_0 to T_4
M = 4.0       # Coefficients are sampled from [-1, 1]
num_samples = 10000  # Number of u(x) samples to generate
nPoints = 100

# Generate samples
sampled_functions, coeffs = sampleChebPoly(degree_N, M, num_samples)

generateDataFromChebFunc1D(sampled_functions,nPoints)