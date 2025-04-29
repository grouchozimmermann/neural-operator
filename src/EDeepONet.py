from config import ConfigEDeepONet
from resources import BaseArchitecture
from layers import FeedForwardNeuralNetwork
import torch
from utils.dataclass import ExtendedDeepONetDataset
from plotting.plotLoss import plotSeparateOutputLoss
from plotting.plotPrediction import plotUnseenPrediction
import numpy as np
import time

class EDeepONetModel(torch.nn.Module):
    def __init__(self, config: ConfigEDeepONet):
        super().__init__()

        self.config = config
        self.enableBias = config.enableFinalBias

        self.inputBranches = torch.nn.ModuleList([FeedForwardNeuralNetwork(config.branchInChannels, config.branchOutChannels, config.branchNumberOfHiddenLayers, config.branchHiddenLayerSize, config.activation, True, False, config.initialization, config) for _ in range(3)])
        self.mainBranch = FeedForwardNeuralNetwork(config.mainBranchInChannels, config.mainBranchOutChannels, config.mainBranchNumberOfHiddenLayers, config.mainBranchHiddenLayerSize, config.activation, True, config.mainBranchActivateLastLayer, config.initialization, config)
        self.trunk = FeedForwardNeuralNetwork(config.trunkInChannels, config.trunkOutChannels, config.trunkNumberOfHiddenLayers, config.trunkHiddenLayerSize, config.activation, True, config.trunkActivateLastLayer, config.initialization, config)
        if self.enableBias:
            self.bias = torch.nn.Parameter(torch.zeros(self.config.numOutput,1))
        
    def forward(self, input, domain):
            
        inputToMain = self.inputBranches[0](input[:,0,:]) * self.inputBranches[1](input[:,1,:]) * self.inputBranches[2](input[:,2,:])
        branchOutput = self.mainBranch(inputToMain)
        trunkOutput = self.trunk(domain)

        numBasisFunctions = int(trunkOutput.shape[2] / self.config.numOutput)
        
        branchOutput_reshaped = branchOutput.view(branchOutput.shape[0], self.config.numOutput, numBasisFunctions)
        trunkOutput_reshaped = trunkOutput.view(trunkOutput.shape[0],trunkOutput.shape[1], self.config.numOutput, numBasisFunctions)

        innerProduct = torch.einsum('sop, stop -> sot', branchOutput_reshaped, trunkOutput_reshaped)
        if self.enableBias:        
            innerProduct = innerProduct + self.bias
        return innerProduct

class EDeepONet(BaseArchitecture): 
    def __init__(self, config:ConfigEDeepONet):
        super().__init__(EDeepONetModel,config)
        self.config = config
        self.LRScheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.90)
        self.terre = torch.empty((config.numOutput,self.config.epochs))
        self.terreTest = torch.empty((config.numOutput,self.config.epochs))

    def setTrainingData(self, input, domain, output):
        self.trainingData = ExtendedDeepONetDataset(input, domain, output)

    def setTestingData(self, input, domain, output):
        self.testingData = ExtendedDeepONetDataset(input, domain, output)

    def predict(self, input, domain, output):
        predTrain = self.model(input, domain)
        loss = self.criterion(predTrain,output)
        return loss


    def train(self, enablePrint = True):
        start = time.time()
        
        input = self.trainingData.forcing_vectors
        domain = self.trainingData.domain[:,:,None]
        output = self.trainingData.output_vectors

        for i in range(self.config.epochs):

            # Reset optimizer
            self.optimizer.zero_grad()

            # Output of the model
            predTrain = self.model(input, domain)

            loss = 0

            for k in range(self.config.numOutput):
               terreLoss = self.relL2Norm(predTrain[:,k,:], output[:,k,:])
               #terreLoss = self.criterion(predTrain[:,k,:], output[:,k,:]) 
               self.terre[k,i] = terreLoss
               loss += terreLoss

            # Calculate the backwards differentiation
            loss.backward()

            # Test clipping gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 120)
                    
            # Step optimzier
            self.optimizer.step()

            # Perform test per epoch
            with torch.no_grad():
                testLoss = 0
                testPrediction = self.model(self.testingData.forcing_vectors, self.testingData.domain[:,:,None])
                for k in range(self.config.numOutput):
                    #aaaaloss = self.criterion(testPrediction[:,k,:], self.testingData.output_vectors[:,k,:]) 
                    aaaaloss = self.relL2Norm(testPrediction[:,k,:], self.testingData.output_vectors[:,k,:]) 
                    self.terreTest[k,i] = aaaaloss
                    testLoss += aaaaloss


            # Learning rate decay
            if self.config.lrScheduler:
                self.LRScheduler.step()

            # Register errors
            self.errorsTrain[i] = loss
            self.errorsTest[i] = testLoss
            self.gradientNorms[i] = self.getGradientNorm()
            self.parameterNorms[i] = self.getParamNorm()
            self.layerGradientNorms[:,i] = self.getLayerParamNorm()

            # Print section
            if (enablePrint and i % 100 == 0):
                self.printTrainingDetails(i)
        
        
        timeTaken = time.time() - start
        if enablePrint:
            print("Training took", timeTaken, "seconds!")
        
        if self.config.saveTrainingData:
            self.saveTrainingSession(self.config, self.model, timeTaken)

    def plotSeparateOutputError(self):
        plotSeparateOutputLoss(self.terre)
    
    def plotUnseenPrediction(self, sample):
        inputSampleToPlot = self.testingData.forcing_vectors[sample,:,:]
        outputSampleToPlot = self.testingData.output_vectors[sample,:,:]
        domainSampleToPlot = self.testingData.domain[sample,:,None]
        plotUnseenPrediction(self.model, inputSampleToPlot[None,:,:], domainSampleToPlot[None,:,:], outputSampleToPlot[None,:,:])

    def plotPrediction(self, sample):
        inputSampleToPlot = self.trainingData.forcing_vectors[sample,:,:]
        outputSampleToPlot = self.trainingData.output_vectors[sample,:,:]
        domainSampleToPlot = self.trainingData.domain[sample,:,None]
        plotUnseenPrediction(self.model, inputSampleToPlot[None,:,:], domainSampleToPlot[None,:,:], outputSampleToPlot[None,:,:])
        

    