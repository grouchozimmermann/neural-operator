from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

targetData = np.genfromtxt(Path("./src/fowt/data_experimental/FOWT1_FW2_Experiment.txt"))
time = targetData[1:,0]
x_com = targetData[1:,1]
y_com = targetData[1:,2]
z_com = targetData[1:,3]
roll = targetData[1:,4]
pitch = targetData[1:,5]
yaw = targetData[1:,6]
t_fore = targetData[1:,7]
t_port = targetData[1:,8]
t_stb = targetData[1:,9]
WG1 = targetData[1:,10]
WG2 = targetData[1:,11]
WG3 = targetData[1:,12]



fig = plt.figure(figsize=(8, 6))
plt.plot(time, WG1, marker='o', linestyle='-', color='b', label='coord vs. time')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Coordinate', fontsize=12)
plt.title('Coordinate vs. Time', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()




