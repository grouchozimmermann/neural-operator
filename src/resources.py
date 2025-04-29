from config import ConfigDeepONet
from layers import CoupledDeepONet as CoupledDeepONetModel
from utils.dataclass import TrainingDataObject
from plotting.plotLoss import plotLoss
from plotting.plotStatisticalErrors import plotStatisticalErrors
from plotting.plotPrediction import plotPrediction, plotPendulumPrediction, plotGResNetPrediction
from plotting.plotGradientNorms import plotGradientNorms
from plotting.plotParameterNorms import plotParameterNorms
from plotting.plotLayerGradNorms import plotLayerGradNorms
from datetime import datetime
import torch
import numpy as np
import time
import os
import json

class BaseArchitecture(): 
    def __init__(self, model, Config:ConfigDeepONet):
        self.config = Config
        self.modelType = model
        self.model = model(Config)
        self.criterion = self.config.lossFunction()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.config.learningRate, weight_decay=self.config.weight_decay)
        

        self.errorsTrain = np.empty([self.config.epochs])
        self.errorsTest = np.empty([self.config.epochs])
        self.gradientNorms = np.empty([self.config.epochs])
        self.parameterNorms = np.empty([self.config.epochs])
        self.layerGradientNorms = np.empty([len(list(self.model.named_parameters())), self.config.epochs])

        self.meanTrainError = 0
        self.meanTestError = 0
        self.stdTrainError = 0
        self.stdTestError = 0

    def meanRelNorm(self, pred,true, epilson = 0.01):
        norm = torch.abs(pred - true)/(torch.abs(true) + epilson)
        relNorm = norm.sum()/norm.numel()
        return relNorm
    
    def relL2Norm(self,pred,true):
        norm = torch.norm(pred - true,p=2)/torch.norm(true,p=2)
        return norm

    def getRandomBatch(self, batchSize):
        maxNumOfSamples = self.trainingData.input.shape[0]
        indices = torch.randint(0,maxNumOfSamples,(batchSize,))
        batchedOutput = self.trainingData.output[indices,:,:]
        batchedInput = self.trainingData.input[indices,:]

        return batchedInput, batchedOutput

    def setTrainingData(self, input, domain, output):
        self.trainingData = TrainingDataObject(input, domain, output)

    def setTestingData(self, input, domain, output):
        self.testingData = TrainingDataObject(input, domain, output)

    def reInitialiseModel(self):
        self.model = self.modelType(self.config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.config.learningRate)

    def plotTrainingErrors(self, isloglog = True):
        plotLoss(self.errorsTest, self.errorsTrain, isloglog)

    def plotPrediction(self):
        plotPrediction(self.model, self.testingData.input, self.testingData.outputDomain[:,None], self.testingData.output)

    def plotCoupledPrediction(self):
        plotPendulumPrediction(self.model, self.testingData.input, self.testingData.outputDomain[:,None], self.testingData.output)
    
    def plotGResNetPrediction(self):
        plotGResNetPrediction(self.model,self.testingData.input, self.testingData.output)

    def plotGradientNorms(self):
        plotGradientNorms(self.gradientNorms)

    def plotParameterNorms(self):
        plotParameterNorms(self.parameterNorms)
    
    def plotLayerParamNorms(self):
        namedLayerParams = dict((param[0], self.layerGradientNorms[i,:]) for (i, param) in enumerate(self.model.named_parameters()))
        plotLayerGradNorms(namedLayerParams)
    
    def plotStatisticalError(self):
        plotStatisticalErrors(self.meanTrainError, self.meanTestError, self.stdTrainError, self.stdTestError)
    
    def printTrainingDetails(self, epoch):
        print('Epoch:', epoch)
        print('Norm of gradient: ', self.getGradientNorm())
        print('Norm of parameters: ', self.getParamNorm())
        print('Training loss:', self.errorsTrain[epoch])
        print('Testing loss:', self.errorsTest[epoch])
        print('\n \n \n')

    def saveTrainingSession(self, ConfigDeepONet, deepONet, timeTaken):
        # Get the current timestamp and format it
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Directory
        dir = f'./training/session_{timestamp}'

        if not os.path.exists(dir):
            os.mkdir(dir)

        # Save weights
        torch.save(deepONet.state_dict(), os.path.join(dir,'weights.pth'))

        # Save errors
        np.save(os.path.join(dir,'trainingErrors'),self.errorsTrain)
        np.save(os.path.join(dir,'testingErrors'),self.errorsTest)
        np.save(os.path.join(dir, 'gradientNorms'),self.gradientNorms)
         
        # Example data
        metadata = {
                "learning_rate": ConfigDeepONet.learningRate,
                "epochs": ConfigDeepONet.epochs,
                "lossCriterion": ConfigDeepONet.lossFunction.__name__,
                'optimizer': ConfigDeepONet.optimizer.__name__,
                'trunkActivation': ConfigDeepONet.trunkActivation,
                "trunkDepth": ConfigDeepONet.trunkNumberOfHiddenLayers,
                "trunkWidth": ConfigDeepONet.trunkHiddenLayerSize,
                "branchActivation": ConfigDeepONet.branchActivation,
                "branchDepth": ConfigDeepONet.branchNumberOfHiddenLayers,
                "branchWidth": ConfigDeepONet.branchHiddenLayerSize,
                'timeTaken': timeTaken
            }

        # Save to JSON file
        with open(os.path.join(dir,'metadata.json'), 'w') as json_file:
            json.dump(metadata, json_file, indent=4)

    def printParameters(self):
        for name, param in self.model.named_parameters():
            print(name, param.size(), param.requires_grad)

    def printGradientNorm(self):
        totalNorm = 0
        for p in self.model.parameters():
            paramNorm = p.grad.data.norm(2)
            totalNorm += paramNorm.item()**2
        print('Norm of gradient: ', totalNorm**(1./2),'\n')

    def getGradientNorm(self):
        totalNorm = 0
        for p in self.model.parameters():
            paramNorm = p.grad.data.norm(2)
            totalNorm += paramNorm.item()**2
        return totalNorm**(1./2)
    
    def getParamNorm(self):
        totalNorm = 0
        for p in self.model.parameters():
            paramNorm = p.data.norm(2)
            totalNorm += paramNorm.item()**2
        return totalNorm**(1./2)
    
    def getLayerParamNorm(self):
        layerParamsNorms = torch.zeros(len(list(self.model.named_parameters())))
        for (i,param) in enumerate(self.model.parameters()):
                layerParamsNorms[i] = param.grad.data.norm(2)
        return layerParamsNorms


