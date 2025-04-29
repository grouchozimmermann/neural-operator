import matplotlib.pyplot as plt
import numpy as np

def plotLayerGradNorms(layerGradNorms):
    
    #Extremely ugly fix
    nIter = layerGradNorms[list(layerGradNorms.keys())[0]].shape[0]

    iterAxis = np.linspace(1,nIter,nIter)

    # Instantiate figure and sub axes
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot on the first axis
    for (layerName, gradNorms) in layerGradNorms.items():
        ax.loglog(iterAxis, gradNorms, label=layerName)
    ax.set_title("Gradient norms vs iterations")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("L2 norm of gradient")
    ax.legend()
    ax.grid(True)

    # Add an overall title for the entire figure
    fig.suptitle("Recorded gradient norms during training")

    # Display the plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Adjust layout to make room for the overall title
    plt.show(block = False)





