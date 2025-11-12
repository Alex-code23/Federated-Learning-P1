import os

import matplotlib.pyplot as plt

# ---------------------- Plotting helpers ----------------------

def plot_results(results_dict, save_file=None, title='Accuracies'):
    plt.figure(figsize=(8,5))
    for label, stats in results_dict.items():
        plt.plot(stats['accs'], label=label)
    plt.xlabel('Communication rounds')
    plt.ylabel('Test accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_file:
        # if folder does not exist, create it
        folder = os.path.dirname(save_file)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_file)
    else:
        plt.show()

def plot_xi_A(stats, save_file=None,title='Heterogeneity and Disturbance'):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(stats['xi'])
    plt.title('Heterogeneity (xi)')
    plt.subplot(1,2,2)
    plt.plot(stats['A'])
    plt.title('Disturbance (A)')
    plt.suptitle(title)
    if save_file:
        # if folder does not exist, create it
        folder = os.path.dirname(save_file)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_file)
    else:
        plt.show()
