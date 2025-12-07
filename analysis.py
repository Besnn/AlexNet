"""
Utility functions for embeddings, feature inversion, and activation analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def get_embeddings(model, dataloader, device):
    """
    Extract embeddings from a model at a specific layer.

    Args:
        model: PyTorch model
        dataloader: DataLoader for input data
        device: Device to run computations on

    Returns:
        embeddings: numpy array of embeddings
        labels: numpy array of labels
        images_list: numpy array of original images
    """
    model.eval()
    embeddings = []
    labels = []
    images_list = []
    with torch.no_grad():
        for images, lbls in dataloader:
            images_list.append(images.detach().cpu())
            images = images.to(device)
            x = model.features(images)
            x = x.view(x.size(0), 256 * 2 * 2)
            embedding = model.classifier[:5](x)
            embeddings.append(embedding.detach().cpu().numpy())
            labels.append(lbls.numpy())
    return np.concatenate(embeddings), np.concatenate(labels), torch.cat(images_list).numpy()


def plot_embeddings(embeddings, labels, images, classes, num_images=500):
    """
    Plot t-SNE visualization of embeddings with image thumbnails.

    Args:
        embeddings: numpy array of embeddings (N, D)
        labels: numpy array of class labels (N,)
        images: numpy array of images (N, C, H, W)
        classes: list of class names
        num_images: number of images to display as thumbnails
    """
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(15, 12))

    colors = plt.cm.get_cmap("tab10", len(classes))
    for i, class_name in enumerate(classes):
        class_indices = np.where(labels == i)[0]
        ax.scatter(embeddings_2d[class_indices, 0], embeddings_2d[class_indices, 1],
                   color=colors(i), label=class_name, alpha=0.5)

    if num_images > len(images):
        num_images = len(images)
    indices = np.random.choice(len(images), num_images, replace=False)

    for i in indices:
        img = images[i]
        img = img / 2 + 0.5
        img = np.transpose(img, (1, 2, 0))

        imagebox = plt.matplotlib.offsetbox.OffsetImage(img, zoom=0.5, cmap='gray')
        ab = plt.matplotlib.offsetbox.AnnotationBbox(imagebox, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                                                      frameon=False, pad=0.0)
        ax.add_artist(ab)

    ax.legend()
    plt.title("t-SNE visualization of embeddings")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.show()


def normalize_map(m):
    """
    Normalize a feature map to [0, 1] range.

    Args:
        m: numpy array feature map

    Returns:
        Normalized feature map
    """
    m = m.astype(np.float32)
    m = m - m.min()
    if m.max() != 0:
        m = m / m.max()
    return m


def feature_inversion_channel(model, target_layer_index, channel_idx, num_iterations=300, learning_rate=0.05,
                              regularization=0.0001):
    """
    Reconstruct an image that maximizes activation of a specific channel.

    Args:
        model: PyTorch model
        target_layer_index: Index of the target layer in model.features
        channel_idx: Index of the channel to maximize
        num_iterations: Number of optimization iterations
        learning_rate: Learning rate for Adam optimizer
        regularization: L2 regularization coefficient

    Returns:
        Inverted image as numpy array (H, W, 3)
    """
    model.eval()
    inverted_image = torch.randn(1, 3, 32, 32, device=next(model.parameters()).device, requires_grad=True)
    optimizer = optim.Adam([inverted_image], lr=learning_rate)

    activations = None

    def hook(module, input, output):
        nonlocal activations
        activations = output

    target_layer = list(model.features.children())[target_layer_index]
    handle = target_layer.register_forward_hook(hook)

    for iteration in range(num_iterations):
        optimizer.zero_grad()
        with torch.no_grad():
            inverted_image.data = torch.clamp(inverted_image.data, -1, 1)

        _ = model.features[:target_layer_index + 1](inverted_image)

        # Maximize specific channel only
        loss = -torch.mean(activations[:, channel_idx, :, :]) + regularization * torch.mean(inverted_image ** 2)
        loss.backward()
        optimizer.step()

    handle.remove()

    inverted_image_np = inverted_image.detach().squeeze(0).cpu().numpy()
    inverted_image_np = np.transpose(inverted_image_np / 2 + 0.5, (1, 2, 0))
    return np.clip(inverted_image_np, 0, 1)


def feature_inversion(model, target_layer_index, num_iterations=500, learning_rate=0.01, regularization=0.0001):
    """
    Reconstruct an image that maximizes activation of a target layer.

    Args:
        model: PyTorch model
        target_layer_index: Index of the target layer in model.features
        num_iterations: Number of optimization iterations
        learning_rate: Learning rate for Adam optimizer
        regularization: L2 regularization coefficient

    Returns:
        Inverted image as numpy array (H, W, 3)
    """
    model.eval()

    # Create a random input image
    inverted_image = torch.randn(1, 3, 32, 32, device=next(model.parameters()).device, requires_grad=True)

    # Optimizer for the image
    optimizer = optim.Adam([inverted_image], lr=learning_rate)

    # Hook to capture activations
    activations = None

    def hook(module, input, output):
        nonlocal activations
        activations = output

    target_layer = list(model.features.children())[target_layer_index]
    handle = target_layer.register_forward_hook(hook)

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # Clamp image to valid range
        with torch.no_grad():
            inverted_image.data = torch.clamp(inverted_image.data, -1, 1)

        # Forward pass
        _ = model.features[:target_layer_index + 1](inverted_image)

        # Loss: maximize mean activation + regularization
        loss = -torch.mean(activations) + regularization * torch.mean(inverted_image ** 2)

        loss.backward()
        optimizer.step()

    handle.remove()

    # Denormalize and return
    inverted_image_np = inverted_image.detach().squeeze(0).cpu().numpy()
    inverted_image_np = np.transpose(inverted_image_np / 2 + 0.5, (1, 2, 0))
    inverted_image_np = np.clip(inverted_image_np, 0, 1)

    return inverted_image_np


def get_top_activating_images(model, dataloader, target_layer_index, num_channels=8, k=5, device='cpu'):
    """
    Find images that maximally activate specific channels in a target layer.

    Args:
        model: PyTorch model
        dataloader: DataLoader for input data
        target_layer_index: Index of the target layer in model.features
        num_channels: Number of channels to analyze
        k: Number of top images to keep per channel
        device: Device to run computations on

    Returns:
        Dictionary {channel_idx: [(image, activation_value), ...]}
    """
    model.eval()
    channel_activations = {}

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)

            # Get activations at target layer
            activation_model = nn.Sequential(*list(model.features.children())[:target_layer_index + 1])
            activations = activation_model(images)  # B, C, H, W

            # Find max activation per channel
            max_activations_per_channel = torch.amax(activations, dim=[0, 2, 3])  # C

            for ch_idx in range(min(num_channels, activations.shape[1])):
                if ch_idx not in channel_activations:
                    channel_activations[ch_idx] = []

                # Store max activation value and corresponding image
                max_val = max_activations_per_channel[ch_idx].item()
                # Get the image that produced this max activation in this batch
                batch_max_idx = torch.argmax(torch.amax(activations[:, ch_idx, :, :], dim=[1, 2]))
                img = images[batch_max_idx].cpu()
                channel_activations[ch_idx].append((img, max_val))

    # Keep only top k per channel
    for ch_idx in channel_activations:
        channel_activations[ch_idx] = sorted(channel_activations[ch_idx],
                                            key=lambda x: x[1],
                                            reverse=True)[:k]

    return channel_activations

