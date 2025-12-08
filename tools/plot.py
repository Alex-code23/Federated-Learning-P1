import os

import matplotlib.pyplot as plt
import numpy as np

# ---------------------- Plotting helpers ----------------------

def plot_results(results_dict, save_file=None, title=None):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    for label, stats in results_dict.items():
        plt.plot(stats['accs'], label=label)
    plt.xlabel('Communication rounds')
    plt.ylabel('Test accuracy')
    plt.legend()
    plt.grid(True)
    plt.title('Test Accuracy')

    plt.subplot(1,2,2)
    for label, stats in results_dict.items():
        plt.plot(stats['losses'], label=label)
    plt.xlabel('Communication rounds')
    plt.ylabel('Test loss')
    plt.legend()
    plt.grid(True)
    plt.title('Test Loss')

    plt.suptitle(title)
    
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


def plot_partitions_aggregators(results_by_partition, partition_list, save_file=None, title=None, show_loss=True):
    """
    Affiche une figure 2xlen(partition_list) (accuracy et loss par partition).
    results_by_partition : dict mapping partition -> dict mapping aggregator -> stats (stats['accs'], stats['losses'], ...)
    partition_list : ordre des partitions à afficher (ex: ['iid','dirichlet','noniid'])
    show_loss : si False n'affiche que la ligne d'accuracy (1xN)
    """
    n_part = len(partition_list)
    n_rows = 2 if show_loss else 1
    fig, axes = plt.subplots(n_rows, n_part, figsize=(5*n_part, 4*n_rows), squeeze=False)

    # Palette simple (matplotlib choisira automatiquement si nombre variable)
    for col, PART in enumerate(partition_list):
        if PART not in results_by_partition:
            continue
        results = results_by_partition[PART]  # dict: agg -> stats

        # Accuracy subplot (row 0)
        ax_acc = axes[0][col]
        for agg, stats in results.items():
            ax_acc.plot(stats['accs'], label=agg)
        ax_acc.set_xlabel('Communication rounds')
        ax_acc.set_ylabel('Test accuracy')
        ax_acc.set_title(f'Accuracy — {PART}')
        ax_acc.grid(True)
        if col == 0:
            # Afficher légende sur le premier plot pour éviter répétitions
            ax_acc.legend(loc='lower right', fontsize='small')

        # Loss subplot (row 1) if demandé
        if show_loss:
            ax_loss = axes[1][col]
            for agg, stats in results.items():
                ax_loss.plot(stats['losses'], label=agg)
            ax_loss.set_xlabel('Communication rounds')
            ax_loss.set_ylabel('Test loss')
            ax_loss.set_title(f'Loss — {PART}')
            ax_loss.grid(True)
            if col == 0:
                ax_loss.legend(loc='upper right', fontsize='small')

    plt.suptitle(title or '')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # laisser de la place pour le suptitle

    if save_file:
        folder = os.path.dirname(save_file)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_file)
        plt.close(fig)
    else:
        plt.show()

def plot_class_accuracy_evolution(results_by_partition, partition_list, save_file=None, title=None, dataset_name='Dataset'):
    """
    Affiche l'évolution de l'accuracy par classe pour chaque partition et chaque agrégateur.
    Une figure est générée par partition.
    """
    for PART in partition_list:
        if PART not in results_by_partition:
            continue
        
        results = results_by_partition[PART]
        n_agg = len(results)
        fig, axes = plt.subplots(1, n_agg, figsize=(5 * n_agg, 5), squeeze=False)
        
        for col, (agg, stats) in enumerate(results.items()):
            ax = axes[0][col]
            if 'per_class_accs' in stats:
                for i, class_accs in enumerate(stats['per_class_accs']):
                    ax.plot(class_accs, label=f'Class {i}')
            ax.set_title(f'{agg} - {PART}')
            ax.set_xlabel('Communication rounds')
            ax.set_ylabel('Test Accuracy per Class')
            ax.grid(True)
            ax.legend(fontsize='x-small')

        fig.suptitle(title or f'Class Accuracy Evolution for {PART} partition on {dataset_name}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if save_file:
            # Ensure the filename is unique for each partition
            base, ext = os.path.splitext(save_file)
            partition_save_file = f"{base}_{PART}{ext}"
            folder = os.path.dirname(partition_save_file)
            if folder and not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)
            plt.savefig(partition_save_file)
            plt.close(fig)
        else:
            plt.show()


def plot_xi_A(stats, save_file=None, title='Heterogeneity, Disturbance and Variance'):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(stats['xi'])
    plt.title('Heterogeneity (xi)')
    plt.xlabel('Communication rounds')

    plt.subplot(1,2,2)
    plt.plot(stats['A'])
    plt.title('Disturbance (A)')
    plt.xlabel('Communication rounds')

    plt.suptitle(title)
    if save_file:
        folder = os.path.dirname(save_file)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_file)
        plt.close()
    else:
        plt.show()

def plot_xi_A_partitions(results_by_partition, partition_list, save_file=None, title=None):
    """
    Affiche 2xlen(partition_list) : xi (haut) et A (bas)
    results_by_partition : dict[partition][aggregator] = stats
    """
    n_part = len(partition_list)
    fig, axes = plt.subplots(3, n_part, figsize=(5*n_part, 12), squeeze=False)

    for col, PART in enumerate(partition_list):
        if PART not in results_by_partition:
            continue
        results = results_by_partition[PART]

        # Ligne 0 = xi
        ax_xi = axes[0][col]
        for agg, stats in results.items():
            if 'xi' in stats:
                ax_xi.plot(stats['xi'], label=agg)
        ax_xi.set_title(f'Heterogeneity (xi) — {PART}')
        ax_xi.set_xlabel('Rounds')
        ax_xi.set_ylabel('xi')
        ax_xi.grid(True)
        if col == 0:
            ax_xi.legend(loc='upper right', fontsize='small')

        # Ligne 1 = A
        ax_A = axes[1][col]
        for agg, stats in results.items():
            if 'A' in stats:
                ax_A.plot(stats['A'], label=agg)
        ax_A.set_title(f'Disturbance (A) — {PART}')
        ax_A.set_xlabel('Rounds')
        ax_A.set_ylabel('A')
        ax_A.grid(True)
        if col == 0:
            ax_A.legend(loc='upper right', fontsize='small')

        # Ligne 2 = Variance
        ax_var = axes[2][col]
        for agg, stats in results.items():
            if 'variance' in stats:
                ax_var.plot(stats['variance'], label=agg)
        ax_var.set_title(f'Variance of messages — {PART}')
        ax_var.set_xlabel('Rounds')
        ax_var.set_ylabel('Variance')
        ax_var.grid(True)
        if col == 0:
            ax_var.legend(loc='upper right', fontsize='small')

    plt.suptitle(title or 'Comparison of xi, A and Variance across partitions')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_file:
        folder = os.path.dirname(save_file)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_file)
        plt.close(fig)
    else:
        plt.show()

# fonctions de tracé avec bande d'incertitude
def plot_mean_ci_partitions(results_mean, results_ci, partition_list, save_file=None, title=None, show_loss=True):
    n_part = len(partition_list)
    n_rows = 2 if show_loss else 1
    fig, axes = plt.subplots(n_rows, n_part, figsize=(5*n_part, 4*n_rows), squeeze=False)

    for col, PART in enumerate(partition_list):
        # Accuracy
        ax_acc = axes[0][col]
        acc_sup, acc_inf = 0,0
        for agg, stats_mean in results_mean[PART].items():
            if 'accs' not in stats_mean:
                continue
            mean = stats_mean['accs']
            ci = results_ci[PART][agg]['accs']
            x = np.arange(len(mean))
            ax_acc.plot(x, mean, label=agg)
            ax_acc.fill_between(x, mean - ci, mean + ci, alpha=0.2)
            acc_sup = max(acc_sup, np.max(mean + ci))
            acc_inf = min(acc_inf, np.min(mean - ci))
        ax_acc.set_xlabel('Communication rounds')
        ax_acc.set_ylabel('Test accuracy')
        ax_acc.set_title(f'Accuracy — {PART}')
        ax_acc.set_ylim(acc_sup - acc_sup * 0.3 , acc_sup)
        ax_acc.grid(True)
        if col == 0:
            ax_acc.legend(loc='lower right', fontsize='small')

        # Loss
        if show_loss:
            ax_loss = axes[1][col]
            for agg, stats_mean in results_mean[PART].items():
                if 'losses' not in stats_mean:
                    continue
                mean = stats_mean['losses']
                ci = results_ci[PART][agg]['losses']
                x = np.arange(len(mean))
                ax_loss.plot(x, mean, label=agg)
                ax_loss.fill_between(x, mean - ci, mean + ci, alpha=0.2)
            ax_loss.set_xlabel('Communication rounds')
            ax_loss.set_ylabel('Test loss')
            ax_loss.set_title(f'Loss — {PART}')
            ax_loss.grid(True)
            if col == 0:
                ax_loss.legend(loc='upper right', fontsize='small')

    plt.suptitle(title or '')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_file:
        folder = os.path.dirname(save_file)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_file)
        plt.close(fig)
    else:
        plt.show()

def plot_xi_A_partitions_mean_ci(results_mean, results_ci, partition_list, save_file=None, title=None):
    n_part = len(partition_list)
    fig, axes = plt.subplots(3, n_part, figsize=(5*n_part, 12), squeeze=False)

    for col, PART in enumerate(partition_list):
        # xi
        ax_xi = axes[0][col]
        for agg, stats_mean in results_mean[PART].items():
            if 'xi' not in stats_mean:
                continue
            mean = stats_mean['xi']
            ci = results_ci[PART][agg]['xi']
            x = np.arange(len(mean))
            ax_xi.plot(x, mean, label=agg)
            ax_xi.fill_between(x, mean - ci, mean + ci, alpha=0.2)
        ax_xi.set_title(f'Heterogeneity (xi) — {PART}')
        ax_xi.set_xlabel('Rounds')
        ax_xi.set_ylabel('xi')
        ax_xi.grid(True)
        if col == 0:
            ax_xi.legend(loc='upper right', fontsize='small')

        # A
        ax_A = axes[1][col]
        for agg, stats_mean in results_mean[PART].items():
            if 'A' not in stats_mean:
                continue
            mean = stats_mean['A']
            ci = results_ci[PART][agg]['A']
            x = np.arange(len(mean))
            ax_A.plot(x, mean, label=agg)
            ax_A.fill_between(x, mean - ci, mean + ci, alpha=0.2)
        ax_A.set_title(f'Disturbance (A) — {PART}')
        ax_A.set_xlabel('Rounds')
        ax_A.set_ylabel('A')
        ax_A.grid(True)
        if col == 0:
            ax_A.legend(loc='upper right', fontsize='small')

        # variance
        ax_var = axes[2][col]
        for agg, stats_mean in results_mean[PART].items():
            if 'variance' not in stats_mean:
                continue
            mean = stats_mean['variance']
            ci = results_ci[PART][agg]['variance']
            x = np.arange(len(mean))
            ax_var.plot(x, mean, label=agg)
            ax_var.fill_between(x, mean - ci, mean + ci, alpha=0.2)
        ax_var.set_title(f'Variance of messages — {PART}')
        ax_var.set_xlabel('Rounds')
        ax_var.set_ylabel('Variance')
        ax_var.grid(True)
        if col == 0:
            ax_var.legend(loc='upper right', fontsize='small')

    plt.suptitle(title or 'Comparison of xi, A and Variance across partitions (mean ± CI)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_file:
        folder = os.path.dirname(save_file)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_file)
        plt.close(fig)
    else:
        plt.show()
