import pandas as pd
import numpy as np
from pathlib import Path
import os


def removeNoiseAtStart(data: pd.DataFrame, cutOff: int):
    closest_idx = (data.Time - cutOff).abs().idxmin()
    nonNoiseData = data.iloc[closest_idx:,:]
    return nonNoiseData

def interpolateNaN(data, enablePrint):
    """
    Interpolate values containing NaN's
    """
    # Extract time values from the first column
    time = data.iloc[:, 0].values
    
    # Process each column (except the first time column)
    for col_idx in range(1, data.shape[1]):
        col_name = data.columns[col_idx]
        signal = data.iloc[:, col_idx].values
        
        # Check for NaN values
        if np.isnan(signal).any():
            nan_count = np.isnan(signal).sum()
            nan_percentage = (nan_count / len(signal)) * 100
            if enablePrint:
                print(f"Warning: Column {col_name} contains {nan_count} NaN values ({nan_percentage:.2f}% of data).")
            
            # Handle NaN values using linear interpolation instead of zeros
            # This preserves the signal characteristics better than using zeros
            valid_indices = ~np.isnan(signal)
            valid_time = time[valid_indices]
            valid_signal = signal[valid_indices]
            
            # Only interpolate if we have enough valid points
            if len(valid_signal) > 1:
                # Interpolate NaN values
                signal = np.interp(time, valid_time, valid_signal)
                if enablePrint:
                    print(f"  • NaN values interpolated using linear interpolation.")
            else:
                # If too few valid points, replace with zeros as fallback
                if enablePrint:
                    print(f"  • Too few valid data points. NaN values could be replaced with zeros, however break instead.")
                #signal = np.nan_to_num(signal)
                break


        data.iloc[:, col_idx] = signal
    return data

def downSampleData(data):
    nrow = 2000
    fluff = data.Time.shape[0] % nrow
    endIndex = data.Time.shape[0] - fluff
    trunc_data_df = data.iloc[:endIndex,:]
    step = trunc_data_df.shape[0] // nrow
    downsampledDatadf = trunc_data_df.iloc[::step].reset_index(drop=True)
    return downsampledDatadf

def otherDownSampleData(data):
    numSoughtPoints = 2000
    numExtraVersions = 10
    requiredEndIndex = data.Time.shape[0] - numExtraVersions
    terre = np.linspace(0, requiredEndIndex, numSoughtPoints).astype(int)
    downsampledDataArray = []
    for i in range(0,numExtraVersions):
        downsampledDatadf = data.iloc[terre + i,:]
        downsampledDataArray.append(downsampledDatadf)

    return downsampledDataArray


def main():
    folder = './src/data/fowt/pandas_csv/'
    #file_names = ['FOWT_T12A125.txt', 'FOWT_T18A125.txt', 'FOWT_T22A125.txt', 'FOWT_T26A125.txt', 'FOWT_T30A125.txt'] #'FOWT_T12A0125.txt', 'FOWT_T18A0125.txt', 'FOWT_T22A0125.txt', 'FOWT_T26A0125.txt', 'FOWT_T30A0125.txt',, "FOWT1_FW1_Experiment.txt", "FOWT1_FW2_Experiment.txt" 
    #file_names = ["FOWT1_FW2_RISE-AAU-SIGMA.txt"]#, "FOWT1_FW2_RISE-AAU-SIGMA.txt"
    file_names = ["FOWT1_FW2_Experiment.txt"] #, "FOWT1_FW2_Experiment.txt"
    #saveFolder = './src/data/fowt/uniform_downsample_new/'
    saveFolder = './src/data/fowt/uniform_downsample/'

    for file_name in file_names:
        datadf = pd.read_csv(folder + file_name)
        datadf = datadf[['Time', 'x', 'z', 'pitch', 'WG1', 'WG2', 'WG3']]
        datadf = datadf.iloc[0:6402]
        cleanDatadf = interpolateNaN(datadf, enablePrint=True)
        #downsampledDataArray = otherDownSampleData(cleanDatadf)
        #for i,datadf in enumerate(downsampledDataArray):
        #    datadf.to_csv(Path(saveFolder+str(i)+'_'+file_name))
        downsampledData = downSampleData(cleanDatadf)
        downsampledData.to_csv(Path(saveFolder+file_name))

main()










