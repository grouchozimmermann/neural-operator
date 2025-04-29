from config import ConfigDeepONet
from layers import DeepONet as DeepONetModel
from utils.dataclass import TrainingDataObject
from plotting.plotLoss import plotLoss
from plotting.plotPrediction import plotPrediction
from plotting.plotGradientNorms import plotGradientNorms
from plotting.plotParameterNorms import plotParameterNorms
from plotting.plotLayerGradNorms import plotLayerGradNorms
from datetime import datetime
import torch
import numpy as np
import time
import os
import json


class DeepONet(): 
    def __init__(self, Config:ConfigDeepONet):
        self.config = Config
        self.model = DeepONetModel(Config)
        self.criterion = self.config.lossFunction()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.config.learningRate, weight_decay=0.0001)
        
        if Config.lrScheduler: 
            self.LRScheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3000, gamma=0.5)

        self.errorsTrain = np.empty([self.config.epochs])
        self.errorsTest = np.empty([self.config.epochs])
        self.loggedGradientNorms = np.empty([self.config.epochs])
        self.loggedParameterNorms = np.empty([self.config.epochs])
        self.loggedLayerGradNorms = np.empty([len(list(self.model.named_parameters())), self.config.epochs])

    def setTrainingData(self, input, domain, output):
        self.trainingData = TrainingDataObject(input, domain, output)

    def setTestingData(self, input, domain, output):
        self.testingData = TrainingDataObject(input, domain, output)

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
                    
            # Step optimzier
            self.optimizer.step()

            # Perform test per epoch
            testPrediction = self.model(testingInput, testingDomain)
            testLoss = self.criterion(testPrediction,testingOutput)

            # Learning rate decay
            if self.config.lrScheduler: 
                self.LRScheduler.step()

            # Register errors
            self.errorsTrain[i] = loss
            self.errorsTest[i] = testLoss
            self.loggedGradientNorms[i] = self.model.getGradientNorm()
            self.loggedParameterNorms[i] = self.model.getParamNorm()
            self.loggedLayerGradNorms[:,i] = self.model.getLayerParamNorm()

            # Print section
            if (enablePrint and i % 100 == 0):
                print('Epoch:', i,)
                print('Norm of gradient: ', self.model.getGradientNorm())
                print('Norm of parameters: ', self.model.getParamNorm())
                print('Training loss:', loss)
                print('Testing loss:', testLoss)
                print('\n \n \n')
        
        timeTaken = time.time() - start
        if enablePrint:
            print("It took", timeTaken, "seconds!")
        
        if ConfigDeepONet.saveTrainingData:
            self.saveTrainingSession(ConfigDeepONet, self.model, self.errorsTrain, self.errorsTest, timeTaken)

    def plotTrainingErrors(self):
        plotLoss(self.errorsTest, self.errorsTrain)

    def plotPrediction(self):
        plotPrediction(self.model, self.testingData.input, self.testingData.outputDomain[:,None], self.testingData.output)

    def plotGradientNorms(self):
        plotGradientNorms(self.loggedGradientNorms)

    def plotParameterNorms(self):
        plotParameterNorms(self.loggedParameterNorms)
    
    def plotLayerParamNorms(self):
        namedLayerParams = dict((param[0], self.loggedLayerGradNorms[i,:]) for (i, param) in enumerate(self.model.named_parameters()))
        plotLayerGradNorms(namedLayerParams)

    def saveTrainingSession(self, ConfigDeepONet,deepONet, trainingErrors, testingErrors, timeTaken):
        # Get the current timestamp and format it
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Directory
        dir = f'./training/session_{timestamp}'

        if not os.path.exists(dir):
            os.mkdir(dir)

        # Save weights
        torch.save(deepONet.state_dict(), os.path.join(dir,'weights.pth'))

        # Save errors
        np.save(os.path.join(dir,'trainingErrors'),trainingErrors)
        np.save(os.path.join(dir,'testingErrors'),testingErrors)
        np.save(os.path.join(dir, 'gradientNorms'),self.loggedGradientNorms)

                
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



