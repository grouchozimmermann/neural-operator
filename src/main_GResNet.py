from pathlib import Path
from GResNet import GResNetModel
from config import ConfigGResNet
from utils.splitTrainTest import splitTrainTest, splitCoupledTrainTest
import numpy as np


# Warning:
# This implementation is a bit haphazard. The data generation function desperately needs to be updated to include the parameters for the harmonic oscillator
# such as the dampening coefficient etc. Currently these needs to be set in the actual GResNetModel code. The code will run even if you forget to update them there which is 
# high risk for getting bad results. Please make sure that you read the code for the GResNetModel and see where parameters are hard coded. If confused please reach out to me.

initialData = np.load(Path("./src/data/state/HarmOscIn100_M1.0_T10_c2_nPoints1000.npy"))
targetData = np.load(Path("./src/data/state/HarmOscOut100_M1.0_T10_c2_nPoints1000.npy"))

ratio = 0.8
shape = initialData.shape
nSamples = shape[0]
trainInit, testInit = initialData[:int(ratio*nSamples),:], initialData[int(ratio*nSamples):,:]
trainTarget, testTarget = targetData[:int(ratio*nSamples),:], targetData[int(ratio*nSamples):,:]


myModel = GResNetModel(ConfigGResNet)
myModel.setTrainingData(trainInit, 0, trainTarget)
myModel.setTestingData(testInit, 0, testTarget)

#myModel.getStatisticalError()
myModel.train()
myModel.plotGradientNorms()
myModel.plotTrainingErrors()
myModel.plotGResNetPrediction(1000)
myModel.plotGResNetPrediction(1000)
myModel.plotGResNetPrediction(1000)







