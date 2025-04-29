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
        lrScheduler = True


        # ===== General Network Settings ===== #
        primaryActFunc = 'Tanh' # 'ReLu', 'LeakyReLU', 'Tanh', 'Softplus'
        initialisation = 'Glorot_normal' # 'Glorot_normal', 'He'
        
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
        trunkNumberOfHiddenLayers = 4
        trunkHiddenLayerSize = 40
        trunkBias = True
        trunkActivateLastLayer = False
        trunkInit = initialisation

        #===== Branch Network Settings ===== #
        branchDiscretizedPoints = m
        branchInChannels = branchDiscretizedPoints * 1 # Should be equal to the number of discretized points for the input function. FIX ACCOMODATE FOR DISCRETZED POINTS LATER
        branchOutChannels = p
        branchActivation = primaryActFunc
        branchNumberOfHiddenLayers = 6
        branchHiddenLayerSize = 40
        branchBias = True
        branchActivateLastLayer = False
        branchInit = initialisation

        # ===== DeepONet Settings ===== #
        DeepONetBias = True