# Purpose: Define all the sublayers of Neural Operators. 
# Author: Groucho Zimmermann
# Date: 28 Oct 2024

import torch.nn as nn
import torch
from config import ConfigDeepONet, ConfigEDeepONet

import torch.nn as nn
class BaseFeedForwardNeuralNetwork(nn.Module):
    def __init__(self, inputChannels, outputChannels, numberOfHiddenLayers, hiddenLayerSize, activationFunction, enableBias, activateLastLayer, initialization):
        super().__init__()

        self.layers = nn.ModuleList()
        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        self.hiddenLayerSize = hiddenLayerSize
        self.numberOfHiddenLayers = numberOfHiddenLayers
        self.activation = activationFunction
        self.weightInitialization = initialization
        self.activateLastLayer = activateLastLayer
        self.enableBias = enableBias
        self.initialization = initialization
    
    def add_layer(self, inputDim, outputDim, activation, initialization, isOutputLayer=False):
        """Adds a single layer to the network."""
        layer = self._get_init_layer(inputDim, outputDim, initialization)
        self.layers.append(layer)
        if not isOutputLayer or self.activateLastLayer:
            self.layers.append(self._get_activation(activation))
    
    def _get_init_layer(self, inputDim, outputDim, initialization):
        layer = nn.Linear(inputDim, outputDim, bias=self.enableBias)
        self._initialize_weights(layer,initialization)
        return layer

    def _initialize_weights(self, layer, initialisation):
        """Initializes weights based on the specified scheme."""
        if initialisation == 'He':
            nn.init.kaiming_normal_(layer.weight)
        elif initialisation == 'Glorot_normal':
            nn.init.xavier_normal_(layer.weight)

    def _get_activation(self, activation):
        """Returns the corresponding activation function."""
        activations = {
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(),
            'Tanh': nn.Tanh(),
            'Softplus': nn.Softplus(),
            'Swish': nn.SiLU()
        }
        return activations.get(activation) 


class FeedForwardNeuralNetwork(BaseFeedForwardNeuralNetwork):
    def __init__(self, inputChannels, outputChannels, numberOfHiddenLayers, hiddenLayerSize, activationFunction, enableBias, activateLastLayer, initialization, config):
        super().__init__(inputChannels, outputChannels, numberOfHiddenLayers, hiddenLayerSize, activationFunction, enableBias, activateLastLayer, initialization)
        self.config = config

        self.add_layer(self.inputChannels, self.hiddenLayerSize, self.activation, self.initialization)

        for _ in range(self.numberOfHiddenLayers):
            self.add_layer(self.hiddenLayerSize,self.hiddenLayerSize,self.activation, self.initialization)

        self.add_layer(self.hiddenLayerSize, self.outputChannels, self.config.activation, self.config.initialization, isOutputLayer=True)

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class MixedFeedForwardNeuralNetwork(BaseFeedForwardNeuralNetwork):
    def __init__(self, inputChannels, outputChannels, numberOfHiddenLayers, hiddenLayerSize, activationFunction, enableBias, activateLastLayer, initialization, config):
        super().__init__(inputChannels, outputChannels, numberOfHiddenLayers, hiddenLayerSize, activationFunction, enableBias, activateLastLayer, initialization)
        self.config = config
        self.cutOff = config.cutOff

        self.add_layer(self.inputChannels, self.hiddenLayerSize, self.activation, self.initialization)

        for i in range(self.numberOfHiddenLayers):
            if (i < self.numberOfHiddenLayers-self.cutOff):
                self.add_layer(self.hiddenLayerSize,self.hiddenLayerSize,self.activation, self.initialization)
            else:
                self.add_layer(self.hiddenLayerSize,self.hiddenLayerSize, self.config.secondActivation, self.config.secondInitialization)

        self.add_layer(self.hiddenLayerSize, self.outputChannels, self.config.secondActivation, self.config.secondInitialization, isOutputLayer=True)

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class SubnetTrunkNetwork(BaseFeedForwardNeuralNetwork):
    def __init__(self, inputChannels, outputChannels, numberOfHiddenLayers, hiddenLayerSize, activationFunction, enableBias, activateLastLayer, initialization, config:ConfigDeepONet):
        super().__init__(inputChannels, outputChannels, numberOfHiddenLayers, hiddenLayerSize, activationFunction, enableBias, activateLastLayer, initialization)
        self.config = config
        self.cutOff = ConfigDeepONet.cutOff

        self.add_layer(self.inputChannels, self.hiddenLayerSize, self.activation, self.initialization)

        for i in range(self.numberOfHiddenLayers):
            if (i < self.numberOfHiddenLayers-self.cutOff):
                self.add_layer(self.hiddenLayerSize,self.hiddenLayerSize,self.activation, self.initialization)
            else:
                if (self.config.secondActivation != 'none'):
                    self.add_layer(self.hiddenLayerSize,self.hiddenLayerSize, self.config.secondActivation, self.config.secondInitialization)
                    
        #self.add_layer(self.hiddenLayerSize, self.outputChannels, self.config.secondActivation, self.config.secondInitialization)
        self.add_layer(self.hiddenLayerSize, self.hiddenLayerSize, self.config.secondActivation, self.config.secondInitialization)

        subnetActivation = self._get_activation(self.activation) if self.config.secondActivation == 'none' else self._get_activation(self.config.secondActivation)
        subnetInitialization = self.initialization if self.config.secondActivation == 'none' else self.config.secondInitialization

        self.subnets = nn.ModuleList([nn.ModuleList([self._get_init_layer(self.hiddenLayerSize,self.hiddenLayerSize, subnetInitialization), subnetActivation,
                                                      self._get_init_layer(self.hiddenLayerSize,self.outputChannels, subnetInitialization), subnetActivation ]) for _ in range(self.config.nSubnets)])


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        subnet_outputs = []
        for i in range(len(self.subnets)):
            x_subnet = x
            for layer in self.subnets[i]:
                x_subnet = layer(x_subnet)
            subnet_outputs.append(x_subnet)
        
        return subnet_outputs    

class CoupledDeepONet(nn.Module): 
    def __init__(self, Config:ConfigDeepONet):
        super().__init__()
        self.config = Config
        self.enableBias = Config.DeepONetBias
        self.trunk = SubnetTrunkNetwork(Config.trunkInChannels, Config.trunkOutChannels, Config.trunkNumberOfHiddenLayers, Config.trunkHiddenLayerSize,activationFunction=Config.trunkActivation, enableBias=Config.trunkBias, activateLastLayer=Config.trunkActivateLastLayer, initialization=Config.trunkInit, config=Config)
        self.branch = FeedForwardNeuralNetwork(Config.branchInChannels, Config.branchOutChannels, Config.branchNumberOfHiddenLayers, Config.branchHiddenLayerSize,Config.branchActivation, enableBias=Config.branchBias, activateLastLayer=Config.branchActivateLastLayer, initialization=Config.branchInit, config=Config)
        if self.enableBias:
            self.bias = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(self.config.nSubnets)])

    def forward(self, u, y):
        u = self.branch(u)
        y = self.trunk(y)

        G1 = torch.einsum('bp,rp  -> br', u, y[0])
        G2 = torch.einsum('bp,rp  -> br', u, y[1])
        if self.enableBias:
            G1 = G1 + self.bias[0]
            G2 = G2 + self.bias[1]
        return torch.stack((G1,G2),1)
    
    def printParameters(self):
        for name, param in self.named_parameters():
            print(name, param.size(), param.requires_grad)

    def printGradientNorm(self):
        totalNorm = 0
        for p in self.parameters():
            paramNorm = p.grad.data.norm(2)
            totalNorm += paramNorm.item()**2
        print('Norm of gradient: ', totalNorm**(1./2),'\n')

    def getGradientNorm(self):
        totalNorm = 0
        for p in self.parameters():
            paramNorm = p.grad.data.norm(2)
            totalNorm += paramNorm.item()**2
        return totalNorm**(1./2)
    
    def getParamNorm(self):
        totalNorm = 0
        for p in self.parameters():
            paramNorm = p.data.norm(2)
            totalNorm += paramNorm.item()**2
        return totalNorm**(1./2)
    
    def getLayerParamNorm(self):
        layerParamsNorms = torch.zeros(len(list(self.named_parameters())))
        for (i,param) in enumerate(self.parameters()):
                layerParamsNorms[i] = param.grad.data.norm(2)
        return layerParamsNorms
    

class DeepONet(nn.Module): 
    def __init__(self, Config:ConfigDeepONet):
        super().__init__()
        self.enableBias = ConfigDeepONet.DeepONetBias
        self.trunk = FeedForwardNeuralNetwork(Config.trunkInChannels, Config.trunkOutChannels, Config.trunkNumberOfHiddenLayers, Config.trunkHiddenLayerSize,activationFunction=Config.trunkActivation, activateLastLayer=Config.trunkActivateLastLayer, initialization=ConfigDeepONet.trunkInit, config=Config, enableBias=Config.DeepONetBias)
        self.branch = FeedForwardNeuralNetwork(Config.branchInChannels, Config.branchOutChannels, Config.branchNumberOfHiddenLayers, Config.branchHiddenLayerSize,Config.branchActivation, activateLastLayer=Config.branchActivateLastLayer, initialization=ConfigDeepONet.branchInit, config=Config, enableBias=Config.DeepONetBias)
        if self.enableBias:
            self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u, y):
        u = self.branch(u)
        y = self.trunk(y)
        
        G = torch.einsum('bp,rp  -> br', u, y)
        if self.enableBias:
            G = G + self.bias
        return G
    
    def printParameters(self):
        for name, param in self.named_parameters():
            print(name, param.size(), param.requires_grad)

    def printGradientNorm(self):
        totalNorm = 0
        for p in self.parameters():
            paramNorm = p.grad.data.norm(2)
            totalNorm += paramNorm.item()**2
        print('Norm of gradient: ', totalNorm**(1./2),'\n')

    def getGradientNorm(self):
        totalNorm = 0
        for p in self.parameters():
            paramNorm = p.grad.data.norm(2)
            totalNorm += paramNorm.item()**2
        return totalNorm**(1./2)
    
    def getParamNorm(self):
        totalNorm = 0
        for p in self.parameters():
            paramNorm = p.data.norm(2)
            totalNorm += paramNorm.item()**2
        return totalNorm**(1./2)
    
    def getLayerParamNorm(self):
        layerParamsNorms = torch.zeros(len(list(self.named_parameters())))
        for (i,param) in enumerate(self.parameters()):
                layerParamsNorms[i] = param.grad.data.norm(2)
        return layerParamsNorms


