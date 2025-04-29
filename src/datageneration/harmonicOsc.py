import os
import numpy as np
from scipy.integrate import solve_ivp
from sampleChebyPoly import sampleChebPoly

def generateHarmonicOscillatorData(sampledFunc,nPoints, dampening, mass, spring):
    T = 1
    t = np.linspace(0,T,nPoints)
    n = len(sampledFunc)

    IC = [0,0]

    inData = np.empty([n, nPoints])
    outData = np.empty([n, 2, nPoints])
    c = dampening
    m = mass
    k = spring
    for i,u in enumerate(sampledFunc):
        #c = np.random.uniform(0, dampening)
        #m = np.random.uniform(0, mass)
        #k = np.random.uniform(0, spring)
        #inData[i,:] = np.concatenate([u(t), [c, m, k]])
        inData[i,:] = u(t)
        terre = lambda t,y: harmonicOscillator(t,y,u,c,m,k)
        outData[i,:,:] = solve_ivp(terre, t_span=[0,T], y0=IC, t_eval=t).y
    
    # Define the target directory and filename
    target_directory = './src/data/harmonic_oscillator'

    # filename for the file you want to save
    inDataFileName = f"HarmOscRHS{num_samples}_M{M}_over_dampened"
    outDataFileName = f"HarmOscSol{num_samples}_M{M}_over_dampened"
    
    # Ensure the directory exists (optional)
    os.makedirs(target_directory, exist_ok=True)

    np.save(os.path.join(target_directory, inDataFileName), inData)
    np.save(os.path.join(target_directory, outDataFileName), outData)
    
def harmonicOscillator(t, s, u, c, m, k):
    t = 2*t - 1
    s1, s2 = s
    ds1 = s2
    ds2 = -k/m*s1 -c/m*ds1 +1/m*u(t)
    return [ds1, ds2]
    


# Example usage:
degree_N = 4  # For example, degree N=5 uses T_0 to T_4
M = 1.0       # Coefficients are sampled from [-1, 1]
num_samples = 10000  # Number of u(x) samples to generate
nPoints = 100
dampening = 4 # Dampening coefficient range
mass = 1 # Mass range
spring = 1 # Spring coefficient range

# Generate samples
sampled_functions, coeffs = sampleChebPoly(degree_N, M, num_samples)

generateHarmonicOscillatorData(sampled_functions,nPoints, dampening, mass, spring)