import numpy as np


preface = "./src/data/fowt/data_experimental/"
path_name = "FOWT1_FW2_Experiment.txt"
targetData = np.genfromtxt(preface+path_name)

# Remove the tension fore, due to NaN
targetData = np.delete(targetData, [2, 4, 6, 7, 8, 9], axis=1)

# Remove the static start
indices = np.where(targetData[:, 0] == 24)[0]
targetData = targetData[indices[0]:,:]

# Remove end, where WG's are equal to nan
indices = np.where(np.isnan(targetData[:, -1]) == True)[0]
targetData = targetData[:indices[0],:]

# Optionally, save the modified data
np.savetxt("./src/data/fowt/data_experimental_edited/"+path_name, targetData)
