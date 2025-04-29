#!/usr/bin/env python3
"""
Time Series Frequency Analysis Script using Non-Uniform FFT

This script analyzes time series data to find dominant frequencies using PyNUFFT.
The script assumes the first column contains time values and analyzes all other columns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pynufft import NUFFT
import torch
from scipy.signal import find_peaks
import os.path
from pathlib import Path
def load_data(file_path):
    """Load the time series data from the specified file path using np.genfromtxt.
    Assumes the dataset has no headers."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
    
    try:
        # Load data using np.genfromtxt which is more flexible with mixed formats
        # Assume no header and auto-detect delimiter
        raw_data = np.genfromtxt(
            file_path,
            delimiter=None,    # Auto-detect delimiter
            skip_header=0,     # No header
            filling_values=np.nan,  # Fill missing values with NaN
            comments='#',      # Skip lines starting with #
            autostrip=True,    # Strip whitespace
            invalid_raise=False  # Don't raise error on invalid lines
        )
        
        # Generate column names: "Time" for first column, "Signal_X" for others
        num_columns = raw_data.shape[1]
        column_names = ["Time"]  # First column is time
        column_names.extend([f"Signal_{i}" for i in range(1, num_columns)])
            
        # Convert to pandas DataFrame for easier handling
        data = pd.DataFrame(raw_data, columns=column_names[:raw_data.shape[1]])
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print("Falling back to pandas methods...")
        
        try:
            # Try pandas methods as fallback
            data = pd.read_csv(file_path)
        except:
            try:
                data = pd.read_csv(file_path, delimiter=None, engine='python')
            except:
                try:
                    data = pd.read_excel(file_path)
                except:
                    data = pd.read_csv(file_path, delim_whitespace=True)
    
    # Check overall data quality
    total_cells = data.size
    nan_cells = data.isna().sum().sum()
    
    if nan_cells > 0:
        print(f"\nData Quality Warning:")
        print(f"  • Dataset contains {nan_cells} NaN values out of {total_cells} cells ({(nan_cells/total_cells)*100:.2f}%)")
        
        # Report columns with highest NaN counts
        col_nan_counts = data.isna().sum()
        cols_with_nans = col_nan_counts[col_nan_counts > 0].sort_values(ascending=False)
        
        if not cols_with_nans.empty:
            print("  • Columns with most NaN values:")
            for col, count in cols_with_nans.items():
                print(f"    - {col}: {count} NaNs ({(count/len(data))*100:.2f}% of column)")
    
    return data

def perform_nufft_analysis(data):
    """
    Perform Non-Uniform FFT analysis on each column of the data.
    Assumes the first column is time and skips it.
    """
    # Extract time values from the first column
    time = data.iloc[:, 0].values
    
    # Normalize time to [0, 2π] for NUFFT
    time_normalized = 2 * np.pi * (time - time.min()) / (time.max() - time.min())
    time_normalized = time_normalized.reshape(-1, 1)  # Reshape to column vector (N, 1)
    
    # Create NUFFT operator
    Nd = (len(time),)  # Size of the output array
    Kd = (len(time)*2,)  # Size of the Fourier domain
    Jd = (6,)  # Number of neighbors to use for interpolation
    
    # Initialize results dictionary
    results = {}
    
    # Process each column (except the first time column)
    for col_idx in range(1, data.shape[1]):
        col_name = data.columns[col_idx]
        signal = data.iloc[:, col_idx].values
        
        # Check for NaN values
        if np.isnan(signal).any():
            nan_count = np.isnan(signal).sum()
            nan_percentage = (nan_count / len(signal)) * 100
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
                print(f"  • NaN values interpolated using linear interpolation.")
            else:
                # If too few valid points, replace with zeros as fallback
                print(f"  • Too few valid data points. NaN values could be replaced with zeros, however break instead.")
                #signal = np.nan_to_num(signal)
                break
                
        
        # Initialize NUFFT object
        nufft = NUFFT()
        nufft.plan(time_normalized, Nd, Kd, Jd)
        
        # Perform forward NUFFT
        spectrum = nufft.forward(signal)
        
        # Calculate frequency axis
        fs = 1 / np.median(np.diff(time))  # Approximate sampling frequency
        freq_axis = np.fft.fftfreq(Kd[0], 1/fs)
        
        # Get magnitude spectrum (only positive frequencies)
        half_point = len(freq_axis) // 2
        pos_freq = freq_axis[:half_point]
        magnitude = np.abs(spectrum[:half_point])
        
        # Find peaks in the spectrum
        peaks, _ = find_peaks(magnitude, height=0.1*np.max(magnitude))
        
        # Sort peaks by magnitude
        peak_magnitudes = magnitude[peaks]
        sorted_indices = np.argsort(peak_magnitudes)[::-1]  # Sort in descending order
        sorted_peaks = peaks[sorted_indices]
        
        # Store results
        results[col_name] = {
            'spectrum': magnitude,
            'frequencies': pos_freq,
            'dominant_peaks': sorted_peaks[:5] if len(sorted_peaks) >= 5 else sorted_peaks,  # Top 5 peaks
            'peak_frequencies': pos_freq[sorted_peaks[:5]] if len(sorted_peaks) >= 5 else pos_freq[sorted_peaks],
            'peak_magnitudes': peak_magnitudes[sorted_indices[:5]] if len(sorted_indices) >= 5 else peak_magnitudes[sorted_indices]
        }
    
    return results

def print_analysis_results(results):
    """Print a nice message with information about each column's frequency content."""
    print("\n" + "="*80)
    print(" "*30 + "FREQUENCY ANALYSIS RESULTS")
    print("="*80 + "\n")
    
    for col_name, result in results.items():
        print(f"\n{'-'*40}")
        print(f"Column: {col_name}")
        print(f"{'-'*40}")
        
        if len(result['dominant_peaks']) > 0:
            highest_freq_idx = 0  # The first peak is already the highest magnitude
            highest_freq = result['peak_frequencies'][highest_freq_idx]
            highest_magnitude = result['peak_magnitudes'][highest_freq_idx]
            
            print(f"  • Highest frequency component: {highest_freq:.4f} Hz")
            print(f"  • Magnitude of highest frequency: {highest_magnitude:.4f}")
            
            if len(result['dominant_peaks']) > 1:
                print("\n  Other significant frequency components:")
                for i in range(1, len(result['peak_frequencies'])):
                    freq = result['peak_frequencies'][i]
                    mag = result['peak_magnitudes'][i]
                    print(f"  • {freq:.4f} Hz (magnitude: {mag:.4f})")
            
            # Calculate ratio of highest to median magnitude
            median_magnitude = np.median(result['spectrum'])
            peak_to_median_ratio = highest_magnitude / median_magnitude if median_magnitude > 0 else float('inf')
            print(f"\n  • Peak-to-median ratio: {peak_to_median_ratio:.2f}")
            
            # Interpret the results
            if peak_to_median_ratio > 10:
                print("  • Interpretation: Strong periodic component detected")
            elif peak_to_median_ratio > 5:
                print("  • Interpretation: Moderate periodic component detected")
            else:
                print("  • Interpretation: Weak or no clear periodic component")
        else:
            print("  • No significant frequency components detected")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80 + "\n")

def main(file_path):
    """Main function to run the analysis."""
    print(f"Analyzing time series data from: {file_path}")
    
    # Load the data
    data = load_data(file_path)
    
    # Print basic information about the dataset
    print(f"\nDataset information:")
    print(f"  • Number of time steps: {data.shape[0]}")
    print(f"  • Number of data columns: {data.shape[1] - 1}")  # Excluding time column
    print(f"  • Column names: {', '.join(data.columns[1:])}")
    
    # Check time column for issues
    time_col = data.iloc[:, 0]
    time_diffs = np.diff(time_col)
    
    # Check for uniformity in time steps
    if len(time_diffs) > 0:
        avg_step = np.mean(time_diffs)
        std_step = np.std(time_diffs)
        cv = std_step / avg_step if avg_step != 0 else float('inf')
        
        if cv > 0.1:  # More than 10% variation in time steps
            print(f"\nTime Column Analysis:")
            print(f"  • Non-uniform time steps detected (coefficient of variation: {cv:.4f})")
            print(f"  • Average time step: {avg_step:.4f}")
            print(f"  • Standard deviation of time steps: {std_step:.4f}")
            print(f"  • Min time step: {np.min(time_diffs):.4f}")
            print(f"  • Max time step: {np.max(time_diffs):.4f}")
            print(f"  • Using non-uniform FFT is appropriate for this dataset.")
        else:
            print(f"\nTime Column Analysis:")
            print(f"  • Time steps are approximately uniform (coefficient of variation: {cv:.4f})")
            print(f"  • Average time step: {avg_step:.4f}")
    
    # Perform NUFFT analysis
    print("\nPerforming non-uniform Fourier transform analysis...")
    results = perform_nufft_analysis(data)
    
    # Print results
    print_analysis_results(results)
    
    return results


file_path = Path("src/data/fowt/numerical/FOWT_T30A125.txt")  # Change this to your actual file path
results = main(file_path)