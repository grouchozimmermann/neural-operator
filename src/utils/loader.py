import torch
import numpy as np
from pathlib import Path


def DataLoader(initialDataFilePath, targetDataFilePath ):
    """
    Given filepaths of both the initial data, and target data, prepares test and validation sets from both.
    The validation and test data are split 80/20 
    """

    # Create Path from string
    initialDataFilePath = Path(initialDataFilePath)
    targetDataFilePath = Path(targetDataFilePath)

    # Load data
    initialData = torch.from_numpy(np.loadtxt(initialDataFilePath)).float()
    targetData = torch.from_numpy(np.loadtxt(targetDataFilePath)).float()

    # Adds the one channel FIX
    initialData = initialData[:,:].to(torch.float64)
    targetData = targetData[:,:].to(torch.float64)

    # Shape of data
    nSamples= initialData.shape[0]

    # Break between validation and test sets.
    breakIndex = int(np.floor(0.8*nSamples))
    
    # Initial data sets
    validInitData = initialData[:breakIndex,:]
    testInitData = initialData[breakIndex:,:]

    # Target data sets
    validTargetData = targetData[:breakIndex,:]
    testTargetData = targetData[breakIndex:,:]

    return validInitData,testInitData, validTargetData, testTargetData





"""pathTest = Path('./src/data/antiderivative_aligned_test.npz')
pathTrain = Path('./src/data/antiderivative_aligned_train.npz')

testData = np.load(pathTest, allow_pickle=True)
trainData = np.load(pathTrain, allow_pickle=True)

validInitData = torch.from_numpy(trainData['X'][0]).to(torch.float64)
validTargetData = torch.from_numpy(trainData['y']).to(torch.float64)
testInitData = torch.from_numpy(testData['X'][0]).to(torch.float64)
testTargetData = torch.from_numpy(testData['y']).to(torch.float64)
y = torch.from_numpy(trainData['X'][1]).to(torch.float64)
"""
