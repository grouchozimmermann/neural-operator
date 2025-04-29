from dataclasses import dataclass, field
from typing import Type, Callable, Optional
import torch.optim as optim
import torch.nn as nn


@dataclass
class ConfigDeepONet:
        # Purpose of this file is to centralize the settings and hyperparameters for a DeepONet Neural Operator.

        # ===== General settings ===== #
        saveTrainingData = True

        # ===== General Network Settings ===== #
        """
        p - Output size of the final layer of the trunk and branch networks
        m - Number of discretized points for the input function
        n -  
        """
        p = 40
        m = 100
        learningRate: float = 0.001
        epochs: int = 20000
        lossFunction = nn.MSELoss  # Storing as type, not instance
        optimizer: Type[optim.Optimizer] = optim.Adam
        lrScheduler = False
        lrStep = 3000
        lrGamma = 0.9
        activation = 'Tanh'


        # ===== General Network Settings ===== #
        primaryActFunc = 'Tanh' # 'ReLu', 'LeakyReLU', 'Tanh', 'Softplus'
        initialization = 'Glorot_normal' # 'Glorot_normal', 'He'
        
        # If using coupled trunk network
        nSubnets = 2
        
        # Multi activation function
        secondActivation = 'Tanh' # 'ReLu', 'LeakyReLU', 'Tanh', 'Softplus', 'none'
        secondInitialization = 'Glorot_normal' # 'Glorot_normal', 'He'
        cutOff = 10

        #===== Trunk Network Settings ===== #       
        trunkInChannels = 1 # Should be equal to dimensionality of a singular point. I.e 3 if 3D etc. 
        trunkOutChannels = p
        trunkActivation = primaryActFunc
        trunkNumberOfHiddenLayers = 2
        trunkHiddenLayerSize = 40
        trunkBias = True
        trunkActivateLastLayer = False
        trunkInit = initialization

        #===== Branch Network Settings ===== #
        branchDiscretizedPoints = m
        branchInChannels = branchDiscretizedPoints * 1 # Should be equal to the number of discretized points for the input function. FIX ACCOMODATE FOR DISCRETZED POINTS LATER
        branchOutChannels = p
        branchActivation = primaryActFunc
        branchNumberOfHiddenLayers = 3
        branchHiddenLayerSize = 40
        branchBias = True
        branchActivateLastLayer = False
        branchInit = initialization

        # ===== DeepONet Settings ===== #
        DeepONetBias = True




@dataclass
class ConfigGResNet:

        # ===== General settings ===== #
        saveTrainingData = True

        # ===== General Network Settings ===== #
        """
        m - Number of discretized points for the input function  
        """
        numOfTimePoints = 1000
        inChannels = 2
        outChannels = 2
        learningRate: float = 0.001
        epochs: int = 8000
        lossFunction = nn.MSELoss  # Storing as type, not instance
        optimizer: Type[optim.Optimizer] = optim.Adam
        lrScheduler = True
        activation = 'Tanh' # 'ReLu', 'LeakyReLU', 'Tanh', 'Softplus'
        initialization = 'Glorot_normal' # 'Glorot_normal', 'He'
        numberOfLayers = 6
        sizeOfLayer = 100
        # FIX
        secondActivation = activation
        secondInitialization = initialization # 'Glorot_normal', 'He'
        cutOff = 0


@dataclass
class ConfigEDeepONet:
        # Purpose of this file is to centralize the settings and hyperparameters for an extended Neural Operator.

        # ===== General settings ===== #
        saveTrainingData = False

        # ===== General Network Settings ===== #
        """
        p - Output size of the final layer of the trunk and branch networks
        m - Number of discretized points for the input function
        n -  
        """
        numOutput = 3
        numInput = 3
        learningRate: float = 0.001
        epochs: int = 60000
        lossFunction = nn.MSELoss  # Storing as type, not instance
        optimizer: Type[optim.Optimizer] = optim.Adam
        lrScheduler = True
        weight_decay = 0.00000

        numBasisFunctions = 50 #Used to be 12

        # ===== Trunk network settings ===== #
        trunkInChannels = 1
        trunkOutChannels = numBasisFunctions * numOutput
        trunkNumberOfHiddenLayers = 6
        trunkHiddenLayerSize = 50
        trunkBias = True
        trunkActivateLastLayer = False

        # ===== First branch network settings ===== #
        branchInChannels = 2000
        branchOutChannels = 50
        branchNumberOfHiddenLayers = 3
        branchHiddenLayerSize = 50
        branchBias = True
        branchActivateLastLayer = False

        # ===== Main branch network settings ===== #
        mainBranchInChannels = branchOutChannels
        mainBranchOutChannels = numBasisFunctions * numOutput
        mainBranchNumberOfHiddenLayers = 5
        mainBranchHiddenLayerSize = 100
        mainBranchBias = True
        mainBranchActivateLastLayer = False

        # ===== General Network Settings ===== #
        activation = 'Tanh' # 'ReLu', 'LeakyReLU', 'Tanh', 'Softplus'
        initialization = 'Glorot_normal' # 'Glorot_normal', 'He'
        
        # ===== DeepONet Settings ===== #
        enableFinalBias = True



