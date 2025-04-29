import matplotlib.pyplot as plt
import numpy as np

def plotLoss(testingErrors, trainingErrors, isloglog=True):
    
    iterAxis = np.linspace(1,testingErrors.shape[0],testingErrors.shape[0])

    # Instantiate figure and sub axes
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot on the first axis
    if isloglog:
        ax.loglog(iterAxis, trainingErrors, label="Training error", color="blue")
        ax.loglog(iterAxis, testingErrors, label="Testing error", color="green")
    else:
        ax.plot(iterAxis, trainingErrors, label="Training error", color="blue")
        ax.plot(iterAxis, testingErrors, label="Testing error", color="green")
    ax.set_title("Error vs iterations")
    ax.set_xlabel("Iterations")
    ax.set_ylabel('Relative MSE')
    ax.legend()
    ax.grid(True)

    # Add an overall title for the entire figure
    fig.suptitle("Comparison of test and training errors")

    # Display the plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Adjust layout to make room for the overall title
    plt.show()

def plotSeparateOutputLoss(errors):
    # Create x-axis values for iterations
    iterAxis = np.linspace(1, errors.shape[1], errors.shape[1])
    names = ['x', 'z', 'pitch']
    
    # Instantiate figure and sub axes
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define a color palette
    colors = ['blue', 'red', 'green']
    
    # Plot each variable with a different color
    for i in range(min(len(names), errors.shape[0])):
        ax.loglog(iterAxis, errors.detach().numpy()[i], label=names[i], color=colors[i])
    
    # Set titles and labels
    ax.set_title("Error vs Iterations")
    ax.set_xlabel("Iterations")
    ax.set_ylabel('Relative MSE')
    ax.legend()
    ax.grid(True)
    
    # Add an overall title for the entire figure
    fig.suptitle("Comparison of Output Errors")
    
    # Display the plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Adjust layout to make room for the overall title
    plt.show()