import pandas as pd
import numpy as np
from pathlib import Path
import os

def load_data(file_path, skipHeader = 0):
    """Load the time series data from the specified file path using np.genfromtxt.
    Assumes the dataset has no headers."""

    # Load data using np.genfromtxt which is more flexible with mixed formats
    # Assume no header and auto-detect delimiter
    raw_data = np.genfromtxt(
            file_path,
            delimiter=None,    # Auto-detect delimiter
            skip_header=skipHeader,     # No header
            filling_values=np.nan,  # Fill missing values with NaN
            comments='#',      # Skip lines starting with #
            autostrip=True,    # Strip whitespace
            invalid_raise=False  # Don't raise error on invalid lines
        )
    

    # ONly for the experimental ones
    #indices = np.where(raw_data[:, 0] == 24)[0]
    #terre = raw_data[indices[0],0] # value of time to be subtracted
    #raw_data = raw_data[indices[0]:,:]
    #raw_data[:,0] = raw_data[:,0] - 24.0
    
    raw_data[:,0] = raw_data[:,0] - raw_data[0,0]
    
    if (raw_data.shape[1] != 13):
        print('Wrong number of columns')
        return
        
    # Column names for the data set
    column_names = ["Time", "x", "y", "z", "roll", "pitch", "yaw", "t_fore", "t_port", "t_stb", "WG1", "WG2", "WG3"]  # First column is time
    
    # Convert to pandas DataFrame for easier handling
    data = pd.DataFrame(raw_data, columns=column_names[:raw_data.shape[1]])

    return data


def main():
    # NOTE: load_data() does require some manual switching when using depending on the sample in question. So be thoughtful when running
    preFolder = './src/data/fowt/numerical_multi_chromatic/'
    #numericalFolder = './src/data/fowt/numerical/'
    #numericalFolderMultiChromatic = './src/data/fowt/numerical_multi_chromatic/'
    #experimentalFolder = "./src/data/fowt/data_experimental/"
    savePreface = './src/data/fowt/pandas_csv/'
    #file_names = ['FOWT_T12A0125.txt', 'FOWT_T18A0125.txt', 'FOWT_T22A0125.txt', 'FOWT_T26A0125.txt', 'FOWT_T30A0125.txt', 'FOWT_T12A125.txt', 'FOWT_T18A125.txt', 'FOWT_T22A125.txt', 'FOWT_T26A125.txt', 'FOWT_T30A125.txt' ]
    #file_names = ["FOWT1_FW1_Experiment.txt", "FOWT1_FW2_Experiment.txt"]
    file_names = ["FOWT1_FW1_RISE-AAU-SIGMA.txt", "FOWT1_FW2_RISE-AAU-SIGMA.txt"]

    for file_name in file_names:
        datadf = load_data(Path(preFolder + file_name)) #
        datadf.to_csv(Path(savePreface+file_name), index=False, na_rep='NaN')

main()









