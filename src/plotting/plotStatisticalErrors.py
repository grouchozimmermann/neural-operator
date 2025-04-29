import matplotlib.pyplot as plt



def plotStatisticalErrors(avgTrainError, avgTestError, stdTrainError, stdTestError):
    
    # Data for plotting
    labels = ['Training', 'Testing']
    means = [avgTrainError,avgTestError]
    stds = [stdTrainError,stdTestError]
    # Create the bar chart
    fig = plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, means, color=['blue', 'orange'], alpha=0.8)

    # Add error bars (symmetric around the mean)
    for bar, mean, std in zip(bars, means, stds):
        x = bar.get_x() + bar.get_width() / 2  # X-position for the error bar
        plt.errorbar(x, mean, yerr=std, capsize=5, fmt='none', color='black', lw=1.5)

    # Add labels to the bars (show mean and std)
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                f'{mean:.2f} ± {std:.2f}',
                ha='center', va='bottom', fontsize=10)

    # Add labels and title
    plt.ylabel('Error', fontsize=12)
    plt.title('Average Error with Standard Deviation', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    #plt.tight_layout()
    plt.show()