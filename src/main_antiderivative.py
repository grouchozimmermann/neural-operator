from DeepONet import DeepONet
from config import ConfigDeepONet
from pathlib import Path
from utils.splitTrainTest import splitTrainTest, splitCoupledTrainTest
import numpy as np


# ===== My data set ===== #
# Comment/uncomment depending on which one you wish to use
initialData = np.load(Path("./src/data/antiderivative/ChebRHS10000.npy"))
targetData = np.load(Path("./src/data/antiderivative/ChebAntiDerivative10000.npy"))

# Used for splitting data into training and testing data.
trainInit, testInit, trainTarget, testTarget = splitTrainTest(initialData, targetData, 0.8)
#trainInit, testInit, trainTarget, testTarget = splitCoupledTrainTest(initialData, targetData, 0.8)

# Create domain, note this implementation is poor. Should be generated alongsinde the data. This is the input for the trunk network,
# and should be equal to the domain on which the target solution is evaluated on. 
y = np.linspace(0,1,100)

# Initialise model
myModel = DeepONet(ConfigDeepONet)
myModel.setTrainingData(trainInit, y, trainTarget)
myModel.setTestingData(testInit, y, testTarget)

myModel.train()
myModel.plotTrainingErrors()
myModel.plotGradientNorms()
myModel.plotPrediction()
myModel.plotPrediction()
myModel.plotPrediction()
myModel.plotPrediction()

