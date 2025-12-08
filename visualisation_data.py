
import torch


def plot_visualisations(trainset, testset, save_dir='./visualisations', dataset_name='MNIST'):
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import Counter

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    def plot_samples(dataset, title, save_path):
        plt.figure(figsize=(8, 8))
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            img, label = dataset[i]
            img = img.permute(1, 2, 0)
            plt.imshow(img.squeeze(), cmap='gray')
            plt.title(f'Label: {label}')
            plt.axis('off')
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_class_distribution(dataset, title, save_path):
        # Correction ici : On utilise .cpu().numpy()
        # .cpu() assure que le tenseur est sur le CPU avant conversion
        # .numpy() est la méthode officielle pour passer de Tensor à Array
        if hasattr(dataset, 'targets'):
             # Cas standard (MNIST, CIFAR...) où les targets sont accessibles directement
            if isinstance(dataset.targets, torch.Tensor):
                targets = dataset.targets.cpu().numpy()
            else:
                targets = np.array(dataset.targets)
        else:
            # Cas de secours si dataset.targets n'existe pas (datasets customs)
            targets = [y for _, y in dataset]
        
        class_counts = Counter(targets)
        # Note : Counter peut renvoyer des clés non triées, c'est mieux de trier pour l'affichage
        classes = sorted(list(class_counts.keys()))
        counts = [class_counts[c] for c in classes]

        plt.figure(figsize=(8, 6))
        plt.bar(classes, counts, color='skyblue')
        plt.xlabel('Class Label')
        plt.ylabel('Number of Samples')
        plt.title(title)
        plt.xticks(classes)
        plt.savefig(save_path)
        plt.close()

    def plot_distrubution_space(dataset, title, save_path):
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        # Extract data and labels
        data = []
        labels = []
        for img, label in dataset:
            data.append(img.numpy().flatten())
            labels.append(label)
        data = np.array(data)
        labels = np.array(labels)

        # Reduce dimensions with PCA
        pca = PCA(n_components=50)
        data_pca = pca.fit_transform(data)

        # Further reduce dimensions with t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        data_tsne = tsne.fit_transform(data_pca)

        # Plotting
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, cmap='tab10', alpha=0.7)
        plt.legend(*scatter.legend_elements(), title="Classes")
        plt.title(title)
        plt.savefig(save_path)
        plt.close()

    # Plot samples from training set
    plot_samples(trainset, f'{dataset_name} Training Set Samples', os.path.join(save_dir, f'{dataset_name}_train_samples.png'))

    # Plot samples from test set
    plot_samples(testset, f'{dataset_name} Test Set Samples', os.path.join(save_dir, f'{dataset_name}_test_samples.png'))

    # Plot class distribution for training set
    plot_class_distribution(trainset, f'{dataset_name} Training Set Class Distribution', os.path.join(save_dir, f'{dataset_name}_train_class_distribution.png'))

    # Plot class distribution for test set
    plot_class_distribution(testset, f'{dataset_name} Test Set Class Distribution', os.path.join(save_dir, f'{dataset_name}_test_class_distribution.png'))

    # Plot data distribution in 2D space for training set
    plot_distrubution_space(trainset, f'{dataset_name} Training Set Data Distribution', os.path.join(save_dir, f'{dataset_name}_train_data_distribution.png'))
    plot_distrubution_space(testset, f'{dataset_name} Test Set Data Distribution', os.path.join(save_dir, f'{dataset_name}_test_data_distribution.png'))

if __name__ == '__main__':
    # load MNIST (assume imports and datasets/transforms disponibles dans ton script principal)
    from torchvision import datasets, transforms

    # DIR = 'visualisation/MNIST'

    # # plot data and class distribution of trainset and testset
    # print("\n --- MNIST ---")
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    plot_visualisations(trainset, testset, save_dir='visualisations', dataset_name='MNIST')

    # others datasets
    # Fashion-MNIST
    DIR = 'visualisation/Fashion-MNIST'
    trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    print("\n --- Fashion-MNIST ---")
    plot_visualisations(trainset, testset, save_dir=DIR, dataset_name='Fashion-MNIST')

    # CIFAR-10
    DIR = 'visualisation/CIFAR-10'
    transform_cifar = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)
    print("\n --- CIFAR-10 ---")
    plot_visualisations(trainset, testset, save_dir=DIR, dataset_name='CIFAR-10')