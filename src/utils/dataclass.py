import torch

class ExtendedDeepONetDataset(torch.utils.data.Dataset):
    """
    Custom dataset for Extended DeepONet where each forcing function and output is a vector
    """
    def __init__(self, forcing_vectors, domain, output_vectors):
        """
        Args:
            forcing_vectors: Tensor of shape [num_samples, num_forcing_functions, vector_length]
                             Each sample has multiple forcing function vectors
            output_vectors: Tensor of shape [num_samples, num_outputs, vector_length_output]
                            Each sample has multiple output vectors
        """
        self.domain = domain
        self.forcing_vectors = forcing_vectors
        self.output_vectors = output_vectors
        
        # Validate data
        assert forcing_vectors.shape[0] == output_vectors.shape[0], "Number of samples must match"
        
    def __len__(self):
        return self.forcing_vectors.shape[0]
    
    def __getitem__(self, idx):
        # Return all forcing vectors and output vectors for this sample
        forcing = self.forcing_vectors[idx]  # Shape: [num_forcing_functions, vector_length]
        output = self.output_vectors[idx]    # Shape: [num_outputs, vector_length_output]
        
        return {'forcing': forcing, 'domain': self.domain,  'output': output}

class TrainingDataObject(torch.utils.data.Dataset):
    """
    TrainingDataObject is a class that extends the PyTorch native dataset class. This is necessary to be usable with 
    Pytorch's DataLoader class. The purpose of the TrainingDataObject class is to aid in passing training data to the network.
    Note: For predicting data use the PredictionDataObject instead.
    
    To function properly with the DeepONet architecture defined in this project
    please structure the passed data as follows.

    Parameters: 
        Both input and output data needs to have the same amount of samples
        - input: The input data to be used for predictions with the model
        - outputDomain: The domain over which predicted output function will be evaluated at (y in Lu Lu's paper)
        - output: The known output to be used for training the network

    dimensionOf(input/output) = (number of samples, spatial points) (This will most likely be expanded as the scope of the input increases)

    I.e for input/output data with 1000 samples and 100 points, it should be structured as follows: (1000, 100)
    """
    def __init__(self, input, outputDomain, output):
        self.input = torch.Tensor(input)
        self.outputDomain = torch.Tensor(outputDomain)
        self.output = torch.Tensor(output)

    def __len__(self):
        return len(self.input)
    
    def __getitem__(self,index):
        item = {
            'input': self.input[index,:],
            'domain': self.outputDomain,
            'output': self.output[index,:]
        }
        return item
class TrainingDataObject(torch.utils.data.Dataset):
    """
    TrainingDataObject is a class that extends the PyTorch native dataset class. This is necessary to be usable with 
    Pytorch's DataLoader class. The purpose of the TrainingDataObject class is to aid in passing training data to the network.
    Note: For predicting data use the PredictionDataObject instead.
    
    To function properly with the DeepONet architecture defined in this project
    please structure the passed data as follows.

    Parameters: 
        Both input and output data needs to have the same amount of samples
        - input: The input data to be used for predictions with the model
        - outputDomain: The domain over which predicted output function will be evaluated at (y in Lu Lu's paper)
        - output: The known output to be used for training the network

    dimensionOf(input/output) = (number of samples, spatial points) (This will most likely be expanded as the scope of the input increases)

    I.e for input/output data with 1000 samples and 100 points, it should be structured as follows: (1000, 100)
    """
    def __init__(self, input, outputDomain, output):
        self.input = torch.Tensor(input)
        self.outputDomain = torch.Tensor(outputDomain)
        self.output = torch.Tensor(output)

    def __len__(self):
        return len(self.input)
    
    def __getitem__(self,index):
        item = {
            'input': self.input[index,:],
            'domain': self.outputDomain,
            'output': self.output[index,:]
        }
        return item
    
class PredictionDataObject(torch.utils.data.Dataset):
    """
    PredictionDataObject is a class that extends the PyTorch native dataset class. This is necessary to be usable with 
    Pytorch's DataLoader class. The purpose of the PredictionDataObject class is to aid in passing input data to the network.
    Note: For training data use the TrainingDataObject instead.
    
    To function properly with the DeepONet architecture defined in this project
    please structure the passed data as follows.

    Parameters: 
        - input: The input data to be used for predictions with the model
        - outputDomain: The domain over which predicted output function will be evaluated at (y in Lu Lu's paper)

    dimensionOf(input) = (number of samples, spatial points) (This will most likely be expanded as the scope of the input increases)

    I.e for input data with 1000 samples and 100 points, it should be structured as follows: (1000, 100)
    """
    def __init__(self,input, outputDomain):
        self.input = torch.Tensor(input)
        self.outputDomain = torch.Tensor(outputDomain)

    def __len__(self):
        return input.shape[0]
    
    def __getitem__(self,index):
        item = {
            'input': self.input[index,:],
            'domain': self.outputDomain,
        }
        return item
    

