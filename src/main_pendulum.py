from DeepONet import DeepONet
from CoupledDeepONet import CoupledDeepONet
from config import ConfigDeepONet
from pathlib import Path
from utils.splitTrainTest import splitTrainTest, splitCoupledTrainTest
import numpy as np


# ===== My data set ===== #
# Comment/uncomment depending on which one you wish to use
initialData = np.load(Path("./src/data/pendulum/PendulumRHS10000_M1.0_N4_n100.npy"))
targetData = np.load(Path("./src/data/pendulum/PendulumSol10000_M1.0_N4_n100.npy"))

# Used for splitting data into training and testing data.
#trainInit, testInit, trainTarget, testTarget = splitTrainTest(initialData, targetData, 0.8)
trainInit, testInit, trainTarget, testTarget = splitCoupledTrainTest(initialData, targetData, 0.8)

# Create domain, note this implementation is poor. Should be generated alongsinde the data. This is the input for the trunk network,
# and should be equal to the domain on which the target solution is evaluated on. 
y = np.linspace(0,1,100)

# Initialise model
myModel = CoupledDeepONet(ConfigDeepONet)
myModel.setTrainingData(trainInit, y, trainTarget)
myModel.setTestingData(testInit, y, testTarget)

myModel.train()
myModel.plotTrainingErrors()
myModel.plotGradientNorms()
myModel.plotCoupledPrediction()
myModel.plotCoupledPrediction()
myModel.plotCoupledPrediction()
myModel.plotCoupledPrediction()

#myModel.getStatisticalError()
