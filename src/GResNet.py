import torch.nn as nn
import torch
import numpy as np
from config import ConfigGResNet
from layers import FeedForwardNeuralNetwork
from CoupledDeepONet import BaseArchitecture
import time

class GResNet(nn.Module):
    def __init__(self, config:ConfigGResNet):
        super().__init__()
        self.config = config
        self.NN = FeedForwardNeuralNetwork(self.config.inChannels, self.config.outChannels, self.config.numberOfLayers, self.config.sizeOfLayer, self.config.activation, True, True, self.config.initialization, self.config)

    def L_euler(self,u):
        #FIX ADD PARAMETERS FOR c, k, m
        T = 10
        c = 2
        m = 1
        k = 1
        dt = T/(self.config.numOfTimePoints-1)
        A = torch.eye(2) + dt*torch.tensor([[0,1], [-k/m, -c/m]])
        return torch.mm(u[:,:2], A.T)

    def L(self,u):
        return u[:,:2]

    def forward(self,x):
        x = self.L_euler(x) + self.NN(x)
        return x

class GResNetModel(BaseArchitecture):
    def __init__(self, config:ConfigGResNet):
        super().__init__(GResNet,config)
        self.config = config

        self.terreTrain = np.empty([int(self.config.epochs/100)])
        self.terreTest = np.empty([int(self.config.epochs/100)])

        self.LRScheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1500, gamma=0.8)
    
    def train(self, enablePrint = True):
        start = time.time()

        input = self.trainingData.input
        output = self.trainingData.output

        testingInput = self.testingData.input
        testingOutput = self.testingData.output

        nTrajectories = 30
        rollout = 5
        latestStart = (self.config.numOfTimePoints-1) - rollout

        inputDataLength = input.shape[0]
        testDataLength = testingInput.shape[0]

        nSamples = int(input.shape[0]/(self.config.numOfTimePoints-1))
        nTestSamples = int(testingInput.shape[0]/(self.config.numOfTimePoints-1))
        
        for i in range(self.config.epochs):

            # Reset optimizer
            self.optimizer.zero_grad()

            # Pick a random state at a point in time, for all samples            
            indices = np.repeat(np.arange(0, inputDataLength, (self.config.numOfTimePoints-1)), nTrajectories) + np.tile(np.random.randint(0,latestStart,nTrajectories),nSamples)
            testIndices = np.repeat(np.arange(0, testDataLength,(self.config.numOfTimePoints-1)),nTrajectories) + np.tile(np.random.randint(0,latestStart,nTrajectories),nTestSamples)

            loss = 0
            testLoss = 0
            
            data = input[indices,:] # Size [indices, 3]
            testData = testingInput[testIndices,:]
            for j in range(rollout):
                prediction = self.model(data)
                loss += self.criterion(prediction,output[indices + j, :])
                data = prediction
                with torch.no_grad():
                    testPrediction = self.model(testData)
                    testLoss += self.criterion(testPrediction,testingOutput[testIndices + j, :])
                    testData = testPrediction

            if (i % 100 == 0):
                totalIndices = np.arange(0, inputDataLength-(self.config.numOfTimePoints-1), (self.config.numOfTimePoints-1))
                testTotalIndices = np.arange(0, testDataLength-(self.config.numOfTimePoints-1), (self.config.numOfTimePoints-1))
                data = input[totalIndices,:]
                testData = testingInput[testTotalIndices,:]
                tempTrainLoss = 0
                tempTestLoss = 0
                for j in range((self.config.numOfTimePoints-1)):
                    with torch.no_grad():
                        prediction = self.model(data)
                        tempTrainLoss += self.criterion(prediction,output[totalIndices + j, :]) 
                        data = prediction
                        #data = torch.cat([prediction, data[:,2:]], dim=1).detach().clone()

                        testPrediction = self.model(testData)
                        tempTestLoss += self.criterion(testPrediction,testingOutput[testTotalIndices + j, :]) 
                        testData = testPrediction
                with torch.no_grad():
                    self.terreTrain[int(i/100)] = tempTrainLoss
                    self.terreTest[int(i/100)] = tempTestLoss
                

            # Calculate the backwards differentiation
            loss.backward()

            # Test clipping gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10**(-1))

            # Step optimzier
            self.optimizer.step()

            # Learning rate decay
            if ConfigGResNet.lrScheduler:
                self.LRScheduler.step()

            # Register errors
            self.errorsTrain[i] = loss
            self.errorsTest[i] = testLoss
            self.gradientNorms[i] = self.getGradientNorm()
            self.parameterNorms[i] = self.getParamNorm()
            self.layerGradientNorms[:,i] = self.getLayerParamNorm()

            # Print section
            if (enablePrint and i % 100 == 0):
                print('Total error at end of training prediction: ', tempTrainLoss )
                print('Total error at end of testing prediction: ', tempTestLoss )
                self.printTrainingDetails(i)
                
        
        
        timeTaken = time.time() - start
        if enablePrint:
            print("Training took", timeTaken, "seconds!")

    def getStatisticalError(self, enablePrint = True):
        numberOfRuns = 5
        multTrainErrors = []
        multTestErrors = []
        multTrainErrorsWhole = []
        multTestErrorsWhole = []
        for i in range(numberOfRuns):
            print('Iteration: ',i)
            self.reInitialiseModel()
            self.train(enablePrint=False)
            multTrainErrors.append(self.errorsTrain[-1])
            multTestErrors.append(self.errorsTest[-1])
            multTrainErrorsWhole.append(self.terreTrain[-1])
            multTestErrorsWhole.append(self.terreTest[-1])
        
        self.meanTrainError = np.mean(multTrainErrors)
        self.meanTestError = np.mean(multTestErrors)
        self.stdTrainError = np.std(multTrainErrors)
        self.stdTestError = np.std(multTestErrors)

        self.meanTrainErrorWhole = np.mean(multTrainErrorsWhole)
        self.meanTestErrorWhole = np.mean(multTestErrorsWhole)
        self.stdTrainErrorWhole = np.std(multTrainErrorsWhole)
        self.stdTestErrorWhole = np.std(multTestErrorsWhole)

        if enablePrint:
            print('\n\n\n')
            print('For ',numberOfRuns,' runs: \n')
            print('Training errors | mean: ', self.meanTrainError, ' | std: ', self.stdTrainError, '\n')
            print('Testing errors | mean: ', self.meanTestError, ' | std: ', self.stdTestError, '\n')
            print('Training errors for whole trajectory | mean: ', self.meanTrainErrorWhole, ' | std: ', self.stdTrainErrorWhole, '\n')
            print('Testing errors for whole trajectory | mean: ', self.meanTestErrorWhole, ' | std: ', self.stdTestErrorWhole, '\n')

        



    