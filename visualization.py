"""
Visualization functions for neural network interpretability.
Includes Grad-CAM, feature maps, and activation visualization.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from analysis import normalize_map, feature_inversion, feature_inversion_channel, get_top_activating_images


def visualize_feature_maps(model, image, target_layer_index, layer_name):
    """
    Visualize feature maps from a specific layer.

    Args:
        model: PyTorch model
        image: Input image tensor (C, H, W)
        target_layer_index: Index of the target layer in model.features
        layer_name: Name of the layer for display
    """
    model.eval()
    activation_model = nn.Sequential(*list(model.features.children())[:target_layer_index + 1])
    image = image.unsqueeze(0).to(next(model.parameters()).device)

    with torch.no_grad():
        activations = activation_model(image)

    activations = activations.squeeze(0).cpu()
    num_feature_maps = activations.shape[0]
    cols = int(np.ceil(np.sqrt(num_feature_maps)))
    rows = int(np.ceil(num_feature_maps / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    fig.suptitle(f'Feature Maps from {layer_name}', fontsize=20)
    for i, ax in enumerate(axes.flat):
        if i < num_feature_maps:
            ax.imshow(activations[i], cmap='viridis')
            ax.set_title(f'Map {i+1}')
        ax.axis('off')
    plt.show()


def generate_grad_cam(model, image, target_class=None):
    """
    Generate Grad-CAM visualization for model interpretation.

    Args:
        model: PyTorch model
        image: Input image tensor (C, H, W)
        target_class: Target class for Grad-CAM (if None, uses predicted class)

    Returns:
        superimposed_img: Superimposed Grad-CAM on original image
        heatmap_resized: Resized heatmap
    """
    model.eval()

    # Disable inplace operations temporarily
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False

    final_conv_layer = None
    for layer in reversed(model.features):
        if isinstance(layer, nn.Conv2d):
            final_conv_layer = layer
            break

    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.clone().detach()

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].clone().detach()

    forward_handle = final_conv_layer.register_forward_hook(forward_hook)
    backward_handle = final_conv_layer.register_full_backward_hook(backward_hook)

    img_tensor = image.unsqueeze(0).to(next(model.parameters()).device)
    img_tensor.requires_grad = True

    output = model(img_tensor)

    if target_class is None:
        target_class = output.argmax(dim=1).item()

    model.zero_grad()
    one_hot_output = torch.zeros_like(output)
    one_hot_output[0][target_class] = 1
    output.backward(gradient=one_hot_output)

    forward_handle.remove()
    backward_handle.remove()

    # Re-enable inplace operations
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = True

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze().detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)

    img_np = image.cpu().numpy()
    img_np = np.transpose(img_np / 2 + 0.5, (1, 2, 0))
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap_color = plt.cm.jet(heatmap_resized)[:, :, :3]
    superimposed_img = (heatmap_color * 0.4) + img_np

    return superimposed_img, heatmap_resized


def visualize_layer_activations_on_real_images(model, dataloader, target_layer_index, layer_name, device='cpu', num_channels=8, k=3):
    """
    Show real dataset images that maximally activate each channel.

    Args:
        model: PyTorch model
        dataloader: DataLoader for input data
        target_layer_index: Index of the target layer in model.features
        layer_name: Name of the layer for display
        device: Device to run computations on
        num_channels: Number of channels to visualize
        k: Number of top images per channel
    """
    top_images = get_top_activating_images(model, dataloader, target_layer_index,
                                          num_channels=num_channels, k=k, device=device)

    fig, axes = plt.subplots(num_channels, k, figsize=(12, 3 * num_channels))
    if num_channels == 1:
        axes = axes.reshape(1, -1)

    for ch_idx, img_list in top_images.items():
        for pos, (img, activation_val) in enumerate(img_list):
            ax = axes[ch_idx, pos]
            img_np = img.numpy()
            img_np = np.transpose(img_np / 2 + 0.5, (1, 2, 0))
            img_np = np.clip(img_np, 0, 1)
            ax.imshow(img_np)
            ax.set_title(f'Ch {ch_idx}, Act: {activation_val:.2f}')
            ax.axis('off')

    plt.suptitle(f'{layer_name}: Real Images with Maximum Activations', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.show()


def visualize_activation_maps_for_image(model, image, target_layer_index, layer_name, image_label=None, num_maps=64):
    """
    Show top activated feature maps for a single image as a grid.

    Args:
        model: PyTorch model
        image: Input image tensor (C, H, W)
        target_layer_index: Index of the target layer in model.features
        layer_name: Name of the layer for display
        image_label: Label for the image (optional)
        num_maps: Number of feature maps to display
    """
    model.eval()
    activation_model = nn.Sequential(*list(model.features.children())[:target_layer_index + 1])
    image_tensor = image.unsqueeze(0).to(next(model.parameters()).device)

    with torch.no_grad():
        activations = activation_model(image_tensor).squeeze(0).cpu().numpy()  # C, H, W

    # Get top num_maps activated channels by mean activation
    mean_activations = np.mean(activations, axis=(1, 2))
    num_channels = min(num_maps, activations.shape[0])
    top_channels = np.argsort(mean_activations)[-num_channels:][::-1]

    # Calculate grid size (8x8 for 64 maps)
    cols = int(np.ceil(np.sqrt(num_channels)))
    rows = int(np.ceil(num_channels / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
    title = f'{layer_name}: Top {num_channels} Activated Feature Maps'
    if image_label is not None:
        title += f' (Image: {image_label})'
    fig.suptitle(title, fontsize=16)

    # Show original image in the first subplot
    ax = axes.flat[0] if rows > 1 else axes[0]
    img_display = np.transpose(image.cpu().numpy() / 2 + 0.5, (1, 2, 0))
    ax.imshow(np.clip(img_display, 0, 1))
    ax.set_title('Original Image', fontweight='bold')
    ax.axis('off')

    # Show feature maps starting from index 1
    for idx, ch in enumerate(top_channels[:num_channels-1]):
        ax = axes.flat[idx + 1] if rows > 1 else axes[idx + 1]
        ax.imshow(normalize_map(activations[ch]), cmap='viridis')
        ax.set_title(f'Ch {ch} (μ={mean_activations[ch]:.2f})', fontsize=8)
        ax.axis('off')

    # Hide any unused subplots
    for idx in range(num_channels, rows * cols):
        axes.flat[idx].axis('off')

    plt.tight_layout()
    plt.show()


def generate_activation_atlas_per_channel(model, layer_indices, num_channels=8):
    """
    Generate atlas showing feature inversion per channel (not per layer).

    Args:
        model: PyTorch model
        layer_indices: List of layer indices to visualize
        num_channels: Number of channels per layer
    """
    num_layers = len(layer_indices)
    fig, axes = plt.subplots(num_layers, num_channels, figsize=(16, 4 * num_layers))

    if num_layers == 1:
        axes = axes.reshape(1, -1)

    for layer_idx, target_layer in enumerate(layer_indices):
        for ch_idx in range(num_channels):
            ax = axes[layer_idx, ch_idx]

            # Invert specific channel
            inverted = feature_inversion_channel(model, target_layer_index=target_layer,
                                                 channel_idx=ch_idx, num_iterations=300,
                                                 learning_rate=0.05)

            ax.imshow(inverted)
            ax.set_title(f'Layer {target_layer}, Ch {ch_idx}', fontsize=9)
            ax.axis('off')

    plt.tight_layout()
    plt.suptitle('Feature Inversion Atlas - Per-Channel Patterns', fontsize=14, y=0.995)
    plt.show()


def generate_activation_atlas(model, layer_indices, grid_size=4):
    """
    Generate an activation atlas showing feature inversions for multiple layers.

    Args:
        model: PyTorch model
        layer_indices: List of layer indices to visualize
        grid_size: Number of features to visualize per layer
    """
    num_layers = len(layer_indices)
    fig, axes = plt.subplots(num_layers, grid_size, figsize=(15, 4 * num_layers))

    if num_layers == 1:
        axes = axes.reshape(1, -1)

    for layer_idx, target_layer in enumerate(layer_indices):
        for feature_idx in range(grid_size):
            ax = axes[layer_idx, feature_idx]

            # Generate feature inversion (synthetic optimal input)
            inverted = feature_inversion(model, target_layer_index=target_layer,
                                         num_iterations=1000, learning_rate=0.1)

            ax.imshow(inverted)
            ax.set_title(f'Layer {target_layer}, Feature {feature_idx}')
            ax.axis('off')

    plt.tight_layout()
    plt.suptitle('Activation Atlas - Feature Inversions (Synthetic Optimal Inputs)', fontsize=16, y=1.001)
    plt.show()


def compare_real_vs_synthetic(model, dataloader, target_layer_index, layer_name, device='cpu', num_channels=4):
    """
    Show real images vs their synthetic inversions side-by-side.

    Args:
        model: PyTorch model
        dataloader: DataLoader for input data
        target_layer_index: Index of the target layer in model.features
        layer_name: Name of the layer for display
        device: Device to run computations on
        num_channels: Number of channels to compare
    """
    top_images = get_top_activating_images(model, dataloader, target_layer_index,
                                           num_channels=num_channels, k=1, device=device)

    fig, axes = plt.subplots(num_channels, 2, figsize=(8, 3 * num_channels))

    for ch_idx in range(num_channels):
        if ch_idx in top_images:
            # Real image
            real_img, _ = top_images[ch_idx][0]
            real_np = real_img.numpy()
            real_np = np.transpose(real_np / 2 + 0.5, (1, 2, 0))
            axes[ch_idx, 0].imshow(np.clip(real_np, 0, 1))
            axes[ch_idx, 0].set_title(f'Real Ch {ch_idx}')
            axes[ch_idx, 0].axis('off')

            # Synthetic inversion
            synthetic = feature_inversion_channel(model, target_layer_index, ch_idx)
            axes[ch_idx, 1].imshow(synthetic)
            axes[ch_idx, 1].set_title(f'Synthetic Ch {ch_idx}')
            axes[ch_idx, 1].axis('off')

    plt.suptitle(f'{layer_name}: Real vs Synthetic Patterns', fontsize=14)
    plt.tight_layout()
    plt.show()

