import numpy as np
def splitTrainTest(initialData, targetData, ratio = 0.8):

    # Shape of data
    nSamples= initialData.shape[0]

    # Break between validation and test sets.
    breakIndex = int(np.floor(ratio*nSamples))
    
    # Initial data sets
    trainingInitData = initialData[:breakIndex,:]
    testingInitData = initialData[breakIndex:,:]

    # Target data sets
    trainingTargetData = targetData[:breakIndex,:]
    testingTargetData = targetData[breakIndex:,:]
    return trainingInitData, testingInitData, trainingTargetData, testingTargetData

def splitCoupledTrainTest(initialData, targetData, ratio = 0.8):

    # Shape of data
    nSamples= initialData.shape[0]

    # Break between validation and test sets.
    breakIndex = int(np.floor(ratio*nSamples))
    
    # Initial data sets
    trainingInitData = initialData[:breakIndex,:]
    testingInitData = initialData[breakIndex:,:]

    # Target data sets
    trainingTargetData = targetData[:breakIndex,:,:]
    testingTargetData = targetData[breakIndex:,:,:]
    return trainingInitData, testingInitData, trainingTargetData, testingTargetData
