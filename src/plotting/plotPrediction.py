import matplotlib.pyplot as plt
import numpy as np
import torch

def plotPrediction(deepONet, testInitData, domain, testTargetData):
    nTests = 4
    indices = [np.random.randint(0,testInitData.shape[0]) for i in range(nTests)]
    testInitData = testInitData[indices,:]
    testTargetData = testTargetData[indices,:]
    predictionData = deepONet(testInitData,domain)
    

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Plot 1: Line plot
    axs[0, 0].plot(domain, predictionData[0,:].detach().numpy(), 'b', label='Prediction')
    axs[0, 0].plot(domain, testTargetData[0,:].detach().numpy(), 'r', label='Analytical')
    axs[0, 0].set_title("Example one")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot 2: Scatter plot
    axs[0, 1].plot(domain, predictionData[1,:].detach().numpy(), color='b', label='Prediction')
    axs[0, 1].plot(domain, testTargetData[1,:].detach().numpy(), color='r', label='Analytical')
    axs[0, 1].set_title('Example two')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot 3: Bar plot
    axs[1, 0].plot(domain, predictionData[2,:].detach().numpy(), color='b', label='Prediction')
    axs[1, 0].plot(domain, testTargetData[2,:].detach().numpy(), color='r', label='Analytical')
    axs[1, 0].set_title('Example three')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot 4: Exponential plot
    axs[1, 1].plot(domain, predictionData[3,:].detach().numpy(), 'b', label='Prediction')
    axs[1, 1].plot(domain, testTargetData[3,:].detach().numpy(), 'r', label='Analytical')
    axs[1, 1].set_title('Example four')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show plot
    plt.show()

def plotPendulumPrediction(deepONet, testInitData, domain, testTargetData):
    nTests = 4
    indices = [np.random.randint(0,testInitData.shape[0]) for i in range(nTests)]
    testInitData = testInitData[indices,:]
    testTargetData = testTargetData[indices,:,:]
    predictionData = deepONet(testInitData,domain)
    

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    titles = ["Example one", "Example two", "Example three", "Example four"]

    for i, ax in enumerate(axs.flat):
        # Plot predictions
        ax.plot(domain, predictionData[i, 0, :].detach().numpy(), 'b--', label='Angle prediction')
        ax.plot(domain, predictionData[i, 1, :].detach().numpy(), 'r--', label='Ang.vel prediction')

        # Plot analytical solutions
        ax.plot(domain, testTargetData[i, 0, :].detach().numpy(), 'b', label='Angle analytical')
        ax.plot(domain, testTargetData[i, 1, :].detach().numpy(), 'r', label='Ang.vel analytical')

        ax.set_title(titles[i])
        ax.legend()
        ax.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show plot
    plt.show()

def plotPendulumPrediction(deepONet, testInitData, domain, testTargetData):
    nTests = 4
    indices = [np.random.randint(0,testInitData.shape[0]) for i in range(nTests)]
    testInitData = testInitData[indices,:]
    testTargetData = testTargetData[indices,:,:]
    predictionData = deepONet(testInitData,domain)
    

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    titles = ["Example one", "Example two", "Example three", "Example four"]

    for i, ax in enumerate(axs.flat):
        # Plot predictions
        ax.plot(domain, predictionData[i, 0, :].detach().numpy(), 'b--', label='Angle prediction')
        ax.plot(domain, predictionData[i, 1, :].detach().numpy(), 'r--', label='Ang.vel prediction')
        ax.set_ylabel('Angular displacement (rad) / Angular velocity (rad/s)')
        ax.set_xlabel('Time (s)')
        # Plot analytical solutions
        ax.plot(domain, testTargetData[i, 0, :].detach().numpy(), 'b', label='Angle analytical')
        ax.plot(domain, testTargetData[i, 1, :].detach().numpy(), 'r', label='Ang.vel analytical')
        
        ax.set_title(titles[i])
        ax.legend()
        ax.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show plot
    plt.show()


def plotGResNetPrediction(model, initData, targetData, nPoints):
    # Picks a random sample
    i = np.random.randint(0,int(initData.shape[0]/nPoints))
    
    # Finds all the indices related to that sample?
    indices = range(i * (nPoints-1), i * (nPoints - 1) + (nPoints - 1))

    initData = initData[indices,:]
    targetData = targetData[indices,:]
    predictionData = model(initData)

    timeAxis = np.linspace(0,10,nPoints)

    # Create a 2x2 grid of subplots
    fig, ax = plt.subplots(1,1, figsize=(10, 8))


    # Plot predictions
    ax.plot(timeAxis[:nPoints-1], predictionData[:, 0].detach().numpy(), 'b--', label='Angle prediction')
    ax.plot(timeAxis[:nPoints-1], predictionData[:, 1].detach().numpy(), 'r--', label='Ang.vel prediction')

    ax.set_ylabel('Angular displacement (rad) / Angular velocity (rad/s)')
    ax.set_xlabel('Time (s)')
    ax.plot(timeAxis[1:], targetData[:,0].detach().numpy(), 'b-', label='Angle analytical')
    ax.plot(timeAxis[1:], targetData[:,1].detach().numpy(), 'r-', label='Ang.vel analytical')

    ax.set_title("Example of prediction")
    ax.legend()
    ax.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Show plot
    plt.show()


def plotUnseenPrediction(deepONet, testInitData, domain, testTargetData):

    predictionData = deepONet(testInitData,domain)
    
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(3, figsize=(10, 8))

    # Plot predictions
    axs[0].plot(domain[0,:,0], predictionData[0, 0, :].detach().numpy(), 'b--', label='predicted x')
    axs[0].plot(domain[0,:,0], testTargetData[0, 0, :].detach().numpy(), 'b', label='computed x')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('x')
    axs[0].set_title('x movement (m)')
    axs[0].legend()
    
    axs[1].plot(domain[0,:,0], predictionData[0, 1, :].detach().numpy(), 'g--', label='predicted z')
    axs[1].plot(domain[0,:,0], testTargetData[0, 1, :].detach().numpy(), 'g', label='computed z')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('z')
    axs[1].set_title('z movement (m)')
    axs[1].legend()

    axs[2].plot(domain[0,:,0], predictionData[0, 2, :].detach().numpy(), 'r--', label='predicted pitch')
    axs[2].plot(domain[0,:,0], testTargetData[0, 2, :].detach().numpy(), 'r', label='computed pitch')
    axs[2].set_ylabel('pitch (rad)')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_title('pitch movement (rad)')
    axs[2].legend()

    fig.suptitle('FOWT movement prediction')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show plot
    plt.show()