from EDeepONet import EDeepONet
from config import ConfigEDeepONet
import numpy as np
import pandas as pd
import torch
from pathlib import Path

preface = './src/data/fowt/uniform_downsample/' #, "FOWT1_FW2_RISE-AAU-SIGMA.txt", "FOWT1_FW1_RISE-AAU-SIGMA.txt"
file_names_train = ['FOWT_T12A125.txt', 'FOWT_T18A125.txt', 'FOWT_T26A125.txt', 'FOWT_T22A125.txt','FOWT_T30A125.txt',"FOWT1_FW2_Experiment.txt", "FOWT1_FW2_RISE-AAU-SIGMA.txt", "FOWT1_FW1_RISE-AAU-SIGMA.txt" ] #, "FOWT1_FW1_Experiment.txt", "FOWT1_FW2_Experiment.txt" 
#file_names_test = ["FOWT1_FW1_Experiment.txt", "FOWT1_FW2_Experiment.txt" ] #, "FOWT1_FW1_Experiment.txt", "FOWT1_FW2_Experiment.txt" 
file_names_test = ["FOWT1_FW1_Experiment.txt"] #"FOWT1_FW2_RISE-AAU-SIGMA.txt", "FOWT1_FW1_RISE-AAU-SIGMA.txt" 

#folder_path = Path(preface)
# Get all files in the folder
#file_names = list(folder_path.glob("*"))

numPoints = ConfigEDeepONet.branchInChannels

input_functions_train = torch.zeros(len(file_names_train),3,numPoints)
output_functions_train = torch.zeros(len(file_names_train),ConfigEDeepONet.numOutput,numPoints)
domain_train = torch.zeros(len(file_names_train),numPoints)


input_functions_test = torch.zeros(len(file_names_test),3,numPoints)
output_functions_test = torch.zeros(len(file_names_test),ConfigEDeepONet.numOutput,numPoints)
domain_test = torch.zeros(len(file_names_test),numPoints)


for i,name in enumerate(file_names_train): 
    targetData = pd.read_csv(preface+name)
    #targetData = pd.read_csv(name)
    input_functions_train[i,:,:] = torch.tensor(targetData[['WG1', 'WG2', 'WG3']].values, dtype=torch.float32).transpose(0,1)
    output_functions_train[i, :, :] = torch.tensor(targetData[['x', 'z', 'pitch']].values, dtype=torch.float32).transpose(0,1)
    domain_train[i, :] = torch.tensor(targetData[['Time']].values, dtype=torch.float32).transpose(0,1)

for i,name in enumerate(file_names_test): 
    targetData = pd.read_csv(preface+name)
    #targetData = pd.read_csv(name)
    input_functions_test[i,:,:] = torch.tensor(targetData[['WG1', 'WG2', 'WG3']].values, dtype=torch.float32).transpose(0,1)
    output_functions_test[i, :, :] = torch.tensor(targetData[['x', 'z', 'pitch']].values, dtype=torch.float32).transpose(0,1)
    domain_test[i, :] = torch.tensor(targetData[['Time']].values, dtype=torch.float32).transpose(0,1)


myModel = EDeepONet(ConfigEDeepONet)
myModel.setTrainingData(input_functions_train, domain_train, output_functions_train)
myModel.setTestingData(input_functions_test, domain_test, output_functions_test)

myModel.train()
myModel.plotTrainingErrors()
myModel.plotSeparateOutputError()
myModel.plotGradientNorms()
myModel.plotPrediction(0)
myModel.plotPrediction(2)
myModel.plotPrediction(5)
myModel.plotUnseenPrediction(0)


