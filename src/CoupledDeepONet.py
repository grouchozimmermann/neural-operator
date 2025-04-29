# The purpose of this file is to define a DeepONet class. 
# Author: Groucho Zimmermann
# Date: 28 Oct 2024

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
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.config.learningRate)
        

        self.errorsTrain = np.empty([self.config.epochs])
        self.errorsTest = np.empty([self.config.epochs])
        self.gradientNorms = np.empty([self.config.epochs])
        self.parameterNorms = np.empty([self.config.epochs])
        self.layerGradientNorms = np.empty([len(list(self.model.named_parameters())), self.config.epochs])

        self.meanTrainError = 0
        self.meanTestError = 0
        self.stdTrainError = 0
        self.stdTestError = 0

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

    def plotTrainingErrors(self):
        plotLoss(self.errorsTest, self.errorsTrain)

    def plotPrediction(self):
        plotPrediction(self.model, self.testingData.input, self.testingData.outputDomain[:,None], self.testingData.output)

    def plotCoupledPrediction(self):
        plotPendulumPrediction(self.model, self.testingData.input, self.testingData.outputDomain[:,None], self.testingData.output)
    
    def plotGResNetPrediction(self, nPoints):
        plotGResNetPrediction(self.model,self.testingData.input, self.testingData.output, nPoints)

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



class CoupledDeepONet(BaseArchitecture): 
    def __init__(self, Config:ConfigDeepONet):
        super().__init__(CoupledDeepONetModel, Config)

        if Config.lrScheduler: 
            self.LRScheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=Config.lrStep, gamma=Config.lrGamma)

    def train(self, enablePrint = True):
        start = time.time()

        input = self.trainingData.input
        domain = self.trainingData.outputDomain[:,None]
        output = self.trainingData.output

        testingInput = self.testingData.input
        testingDomain = self.testingData.outputDomain[:,None]
        testingOutput = self.testingData.output
        
        for i in range(self.config.epochs):

            # Reset optimizer
            self.optimizer.zero_grad()

            # Output of the model
            predTrain = self.model(input, domain)

            # Evaluate loss function for validation
            loss = self.criterion(predTrain,output) 

            # Calculate the backwards differentiation
            loss.backward()

            # Test clipping gradients
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    
            # Step optimzier
            self.optimizer.step()

            # Perform test per epoch
            with torch.no_grad():
                testPrediction = self.model(testingInput, testingDomain)
                testLoss = self.criterion(testPrediction,testingOutput)

            # Learning rate decay
            if ConfigDeepONet.lrScheduler:
                self.LRScheduler.step()

            # Register errors
            self.errorsTrain[i] = loss
            self.errorsTest[i] = testLoss
            self.gradientNorms[i] = self.model.getGradientNorm()
            self.parameterNorms[i] = self.model.getParamNorm()
            self.layerGradientNorms[:,i] = self.model.getLayerParamNorm()

            # Print section
            if (enablePrint and i % 100 == 0):
                self.printTrainingDetails(i)
        
        
        timeTaken = time.time() - start
        if enablePrint:
            print("Training took", timeTaken, "seconds!")
        
        if ConfigDeepONet.saveTrainingData:
            self.saveTrainingSession(ConfigDeepONet, self.model, timeTaken)

    def getStatisticalError(self, enablePrint = True, enableGraph = True):
        numberOfRuns = 10
        multTrainErrors = []
        multTestErrors = []
        for i in range(numberOfRuns):
            print('Iteration: ',i)
            self.reInitialiseModel()
            self.train(enablePrint=False)
            multTrainErrors.append(self.errorsTrain[-1])
            multTestErrors.append(self.errorsTest[-1])
        
        self.meanTrainError = np.mean(multTrainErrors)
        self.meanTestError = np.mean(multTestErrors)
        self.stdTrainError = np.std(multTrainErrors)
        self.stdTestError = np.std(multTestErrors)

        if enablePrint:
            print('\n\n\n')
            print('For ',numberOfRuns,' runs: \n')
            print('Training errors | mean: ', self.meanTrainError, ' | std: ', self.stdTrainError, '\n')
            print('Testing errors | mean: ', self.meanTestError, ' | std: ', self.stdTestError, '\n')

        if enableGraph:
            self.plotStatisticalError()

            

        