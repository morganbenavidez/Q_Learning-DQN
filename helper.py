
import matplotlib.pyplot as plt
import numpy as np
import os



# These stay constant through every experiment
def get_parameters():

    # Parameters for Q-Learning
    learning_rate = 0.1
    discount_factor = 0.99
    # Exploration rate
    epsilon = 0.1

    return learning_rate, discount_factor, epsilon

def build_bins():

    num_bins = 10
    bins = [
        np.linspace(-4.8, 4.8, num_bins),   
        np.linspace(-4, 4, num_bins),       
        np.linspace(-0.418, 0.418, num_bins), 
        np.linspace(-4, 4, num_bins)
    ]
    return num_bins, bins



# Function to plot and save the graphs
def plot_and_save_comparison(ql_data, dqn_data, metric_name, file_path, y_label_name, file_name):

    plt.figure()
    plt.plot(ql_data, label='QL', marker='o')
    plt.plot(dqn_data, label='DQN', marker='^')
    plt.title(f'{metric_name}')
    plt.xlabel('Episodes')
    plt.ylabel(y_label_name)
    plt.legend()
    plt.grid(True)

    # Check if the directory exists; if not, create it
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # Save the figure
    plt.savefig(os.path.join(file_path, f'{file_name}_comparison.png'))
    plt.close()



# Function to plot and save the graphs
def plot_and_save_comparison_error_bars(ql_data, dqn_data, metric_name, file_path, y_label_name, file_name):

    # Calculate mean, min, and max for QL and DQN data
    ql_means = [np.mean(data) for data in ql_data]
    ql_mins = [np.min(data) for data in ql_data]
    ql_maxs = [np.max(data) for data in ql_data]

    dqn_means = [np.mean(data) for data in dqn_data]
    dqn_mins = [np.min(data) for data in dqn_data]
    dqn_maxs = [np.max(data) for data in dqn_data]

    # Errors for QL and DQN
    ql_errors = [np.array([means - mins, maxs - means]) for means, mins, maxs in zip(ql_means, ql_mins, ql_maxs)]
    dqn_errors = [np.array([means - mins, maxs - means]) for means, mins, maxs in zip(dqn_means, dqn_mins, dqn_maxs)]

    # Convert errors into a format suitable for errorbar (2xN arrays)
    ql_errors = np.array(ql_errors).T
    dqn_errors = np.array(dqn_errors).T

    # Plotting
    plt.figure()
    plt.errorbar(range(len(ql_means)), ql_means, yerr=ql_errors, label='QL', fmt='-o', capsize=5)
    plt.errorbar(range(len(dqn_means)), dqn_means, yerr=dqn_errors, label='DQN', fmt='-^', capsize=5)
    plt.title(f'{metric_name}')
    plt.xlabel('Episodes')
    plt.ylabel(y_label_name)
    plt.legend()
    plt.grid(True)

    # Check if the directory exists; if not, create it
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # Save the figure
    plt.savefig(os.path.join(file_path, f'{file_name}_comparison.png'))
    plt.close()



def plot_and_save_comparison_average(ql_data, dqn_data, metric_name, file_path, y_label_name, file_name):
    
    # Calculate the mean for each set of data in QL and DQN
    ql_means = [np.mean(data) for data in ql_data]
    dqn_means = [np.mean(data) for data in dqn_data]

    # Plotting
    plt.figure()
    plt.plot(ql_means, label='QL', marker='o', linestyle='-', color='blue')
    plt.plot(dqn_means, label='DQN', marker='^', linestyle='-', color='red')
    plt.title(f'{metric_name}')
    plt.xlabel('Episodes')
    plt.ylabel(y_label_name)
    plt.legend()
    plt.grid(True)

    # Check if the directory exists; if not, create it
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # Save the figure
    plt.savefig(os.path.join(file_path, f'{file_name}_comparison.png'))
    plt.close()