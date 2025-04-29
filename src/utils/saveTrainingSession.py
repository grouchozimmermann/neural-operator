from datetime import datetime
import torch
import numpy as np
import os
import json


def saveTrainingSession(ConfigDeepONet,deepONet, trainingErrors, testingErrors, timeTaken):
    # Get the current timestamp and format it
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Directory
    dir = f'./training/session_{timestamp}'

    # Ensure the directory exists, create it if not
    trainDirectory = os.makedirs(dir, exist_ok=True)

    # Save weights
    torch.save(deepONet.state_dict(), os.path.join(dir,'weights.pth'))

    # Save errors
    np.savetxt(os.path.join(dir,'trainingErrors.txt'),trainingErrors)
    np.savetxt(os.path.join(dir,'testingErrors.txt'),testingErrors)

            
    # Example data
    metadata = {
            "learning_rate": ConfigDeepONet.learningRate,
            "epochs": ConfigDeepONet.epochs,
            "lossCriterion": ConfigDeepONet.lossFunction.__name__,
            'optimizer': ConfigDeepONet.optimizer.__name__,
            "trunkDepth": ConfigDeepONet.trunkNumberOfHiddenLayers,
            "trunkWidth": ConfigDeepONet.trunkHiddenLayerSize,
            "branchDepth": ConfigDeepONet.branchNumberOfHiddenLayers,
            "branchWidth": ConfigDeepONet.branchHiddenLayerSize,
            'timeTaken': timeTaken
        }

    # Save to JSON file
    with open(os.path.join(dir,'metadata.json'), 'w') as json_file:
        json.dump(metadata, json_file, indent=4)