import matplotlib.pyplot as plt
import numpy as np

def plotParameterNorms(parameterNorms):
    
    nIter = parameterNorms.shape[0]

    iterAxis = np.linspace(1,nIter,nIter)

    # Instantiate figure and sub axes
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot on the first axis
    ax.plot(iterAxis, parameterNorms, label="Parameter norms", color="blue")
    ax.set_title("Parameter norms vs iterations")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("L2 norm of parameter")
    ax.legend()
    ax.grid(True)

    # Add an overall title for the entire figure
    fig.suptitle("Recorded parameter norms during training")

    # Display the plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Adjust layout to make room for the overall title
    plt.show()






