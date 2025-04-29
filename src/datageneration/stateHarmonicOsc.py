import os
import numpy as np
from scipy.integrate import solve_ivp
from sampleChebyPoly import sampleChebPoly

def generateHarmonicOscillatorData(num_samples, sampledFunc,timePoints, dampening, mass, spring):
    T = 10
    t = np.linspace(0,T,timePoints)

    forcing_function = sampledFunc[0]
    inData = np.empty([num_samples*(timePoints-1), 2]) # IC's
    outData = np.empty([num_samples*(timePoints-1), 2])
    c = dampening
    m = mass
    k = spring

    terre1 = timePoints-1
    for i in range(num_samples):
        IC = np.random.uniform(-1,1,2)
        terre = lambda t,y: harmonicOscillator(t,y,forcing_function,c,m,k)
        sol = solve_ivp(terre, t_span=[0,T], y0=IC, t_eval=t).y
        indices = range(i * terre1, i * terre1 + terre1)
        #inData[indices,:] = np.vstack((sol[:,:-1],t[None,:][:,:-1])).T
        inData[indices,:] = sol[:,:-1].T
        outData[indices,:] = sol[:,1:].T
    
    # Define the target directory and filename
    target_directory = './src/data/state'

    # filename for the file you want to save
    inDataFileName = f"HarmOscIn{num_samples}_M{M}_T{T}_c{c}_nPoints{timePoints}"
    outDataFileName = f"HarmOscOut{num_samples}_M{M}_T{T}_c{c}_nPoints{timePoints}"
    functionFileName = f"HarmOscFunc{num_samples}_M{M}_T{T}_c{c}_nPoints{timePoints}"
    
    # Ensure the directory exists (optional)
    os.makedirs(target_directory, exist_ok=True)

    np.save(os.path.join(target_directory, inDataFileName), inData)
    np.save(os.path.join(target_directory, outDataFileName), outData)
    np.save(os.path.join(target_directory, functionFileName), forcing_function)

def harmonicOscillator(t, s, u, c, m, k):
    t = 2*t - 1
    s1, s2 = s
    ds1 = s2
    ds2 = -k/m*s1 -c/m*ds1 +1/m*u(s2)
    return [ds1, ds2]
    
# Example usage:
degree_N = 4  # For example, degree N=5 uses T_0 to T_4
M = 1.0       # Coefficients are sampled from [-1, 1]
num_samples = 100  # Number of initial states to use
nPoints = 1000
dampening = 2 # Dampening coefficient range
mass = 1 # Mass range
spring = 1 # Spring coefficient range

# Generate samples
sampled_functions, coeffs = sampleChebPoly(degree_N, M, 1)

generateHarmonicOscillatorData(num_samples, sampled_functions, nPoints, dampening, mass, spring)