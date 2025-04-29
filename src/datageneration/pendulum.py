import os
import numpy as np
from scipy.integrate import solve_ivp
from sampleChebyPoly import sampleChebPoly

def generatePendulumData(sampledFunc,nPoints):
    """
    Creates and saves two files with input and output data for use in training the DeepONet architecture. 
    The input data will be generated from Chebyshev Polynomial functions, the output data will be the antiderivate of said data.
    Antiderivatives will be solved by RK45

    Parameters: 
    - sampledFunc (list): List of np Chebyshev functions
    - nPoints (int): Number of discretized points
    """
    T = 1
    t = np.linspace(0,T,nPoints)
    n = len(sampledFunc)
    rescaled = True # Rescales the input to the forcing function to accurately work with the Chebyshev polynomial

    IC = [0,0]

    inData = np.empty([n, nPoints])
    outData = np.empty([n, 2, nPoints])

    for i,u in enumerate(sampledFunc):
        inData[i,:] = u(t)
        terre = lambda t,y: pendulum(t,y,u)
        outData[i,:,:] = solve_ivp(terre, t_span=[0,T], y0=IC, t_eval=t).y
    
    # Define the target directory and filename
    target_directory = './src/data/pendulum'

    # filename for the file you want to save
    inDataFileName = f"PendulumRHS{num_samples}_M{M}_N{degree_N}_n{nPoints}"
    outDataFileName = f"PendulumSol{num_samples}_M{M}_N{degree_N}_n{nPoints}"
    
    # Ensure the directory exists (optional)
    os.makedirs(target_directory, exist_ok=True)

    np.save(os.path.join(target_directory, inDataFileName), inData)
    np.save(os.path.join(target_directory, outDataFileName), outData)
    
def pendulum(t, s, u):
    s1, s2 = s
    ds1 = s2
    t = 2*t - 1
    ds2 = -np.sin(s1) + u(t)
    return [ds1, ds2]
    
# Example usage:
degree_N = 4  # For example, degree N=5 uses T_0 to T_4
M = 1.0       # Coefficients are sampled from [-1, 1]
num_samples = 10000  # Number of u(x) samples to generate
nPoints = 100

# Generate samples
sampled_functions, coeffs = sampleChebPoly(degree_N, M, num_samples)

generatePendulumData(sampled_functions,nPoints)