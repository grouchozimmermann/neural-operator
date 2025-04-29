from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def checkNaNs(data):
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


#path_name = "./src/data/fowt/numerical_multi_chromatic/FOWT1_FW2_RISE-AAU-SIGMA.txt"
#targetData = np.genfromtxt(path_name)


path_name = "./src/data/fowt/uniform_downsample/FOWT_T12A125.txt" #FOWT_T30A0125.txt" # FOWT1_FW1_Experiment.txt
#path_name = "./src/data/fowt/pandas_csv/FOWT_T26A125.txt" #FOWT_T30A0125.txt" #FOWT1_FW2_Experiment.txt
targetData = pd.read_csv(path_name)
#checkNaNs(targetData)
targetData = targetData.to_numpy()

targetData = np.delete(targetData,0,1)

time = targetData[:,0]
x_com = targetData[:,1]
z_com = targetData[:,2]
pitch = targetData[:,3]
WG1 = targetData[:,4]
WG2 = targetData[:,5]
WG3 = targetData[:,6]

"""
time = targetData[:,0]
x_com = targetData[:,1]
y_com = targetData[:,2]
z_com = targetData[:,3]
roll = targetData[:,4]
pitch = targetData[:,5]
yaw = targetData[:,6]
t_fore = targetData[:,7]
t_port = targetData[:,8]
t_stb = targetData[:,9]
WG1 = targetData[:,10]
WG2 = targetData[:,11]
WG3 = targetData[:,12]
"""

figs, axs = plt.subplots(3, figsize = (12,10))
axs[0].plot(time, x_com, 'b-', label='x')
axs[0].plot(time, z_com, 'g--', label='z')
#axs[0].set_title('Translational')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Displacement (m)')
axs[0].legend()

axs[1].plot(time, pitch, 'r--', label='Pitch')
#axs[1].set_title('Rotational')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Rotation (rad)')
axs[1].legend()

axs[2].plot(time, WG1, 'b-', label='WG1')
axs[2].plot(time, WG2, 'r--', label='WG2')
axs[2].plot(time, WG3, 'g--', label='WG3')
#axs[2].set_title('Wave forms')
axs[2].set_ylabel('Incident waves (m)')
axs[2].set_xlabel('Time (s)')
axs[2].legend()
plt.suptitle('Translational, rotational and wave form graphs')

plt.show()
