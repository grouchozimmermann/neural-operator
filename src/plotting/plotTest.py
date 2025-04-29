import matplotlib.pyplot as plt
import numpy as np

# Example data
experiments = ['Experiment 1', 'Experiment 2', 'Experiment 3']
training_avg_error = [0.2, 0.15, 0.1]  # Average errors for training data
training_std_dev = [0.05, 0.04, 0.02]  # Standard deviation for training data
testing_avg_error = [0.25, 0.2, 0.18]  # Average errors for testing data
testing_std_dev = [0.06, 0.05, 0.03]  # Standard deviation for testing data

x = np.arange(len(experiments))  # X positions for the groups
width = 0.35  # Width of the bars

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Add bars for training and testing data
training_bars = ax.bar(x - width/2, training_avg_error, width, yerr=training_std_dev, label='Training', capsize=5, color='skyblue')
testing_bars = ax.bar(x + width/2, testing_avg_error, width, yerr=testing_std_dev, label='Testing', capsize=5, color='salmon')

# Add labels, title, and legend
ax.set_xlabel('Experiments')
ax.set_ylabel('Average Error')
ax.set_title('Average Error and Standard Deviation by Experiment')
ax.set_xticks(x)
ax.set_xticklabels(experiments)
ax.legend()

# Add value annotations on the bars (optional)
for bars in [training_bars, testing_bars]:
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom')

# Show the plot
plt.tight_layout()
plt.show()
