import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import cv2

# Define the AlexNet model
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x

def get_embeddings(model, dataloader, device):
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
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(15, 12))
    
    colors = plt.cm.get_cmap("tab10", len(classes))
    for i, class_name in enumerate(classes):
        class_indices = np.where(labels == i)[0]
        ax.scatter(embeddings_2d[class_indices, 0], embeddings_2d[class_indices, 1], color=colors(i), label=class_name, alpha=0.5)

    if num_images > len(images):
        num_images = len(images)
    indices = np.random.choice(len(images), num_images, replace=False)

    for i in indices:
        img = images[i]
        img = img / 2 + 0.5 
        img = np.transpose(img, (1, 2, 0))
        
        imagebox = plt.matplotlib.offsetbox.OffsetImage(img, zoom=0.5, cmap='gray')
        ab = plt.matplotlib.offsetbox.AnnotationBbox(imagebox, (embeddings_2d[i, 0], embeddings_2d[i, 1]), frameon=False, pad=0.0)
        ax.add_artist(ab)
    
    ax.legend()
    plt.title("t-SNE visualization of embeddings")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.show()

def visualize_feature_maps(model, image, target_layer_index, layer_name):
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


def feature_inversion_channel(model, target_layer_index, channel_idx, num_iterations=300, learning_rate=0.05,
                              regularization=0.0001):
    """
    Reconstruct an image that maximizes activation of a specific channel.
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
    Returns the inverted image.
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


def normalize_map(m):
    """Normalize a feature map to [0, 1]."""
    m = m.astype(np.float32)
    m = m - m.min()
    if m.max() != 0:
        m = m / m.max()
    return m


def get_top_activating_images(model, dataloader, target_layer_index, num_channels=8, k=5, device='cpu'):
    """
    Find images that maximally activate specific channels in a target layer.
    Returns: dict {channel_idx: [(image, activation_value), ...]}
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


def visualize_layer_activations_on_real_images(model, dataloader, target_layer_index, layer_name, device='cpu', num_channels=8, k=3):
    """
    Show real dataset images that maximally activate each channel.
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
    layer_indices: list of layer indices to visualize
    grid_size: number of features to visualize per layer
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
                                         num_iterations=300, learning_rate=0.05)

            ax.imshow(inverted)
            ax.set_title(f'Layer {target_layer}, Feature {feature_idx}')
            ax.axis('off')

    plt.tight_layout()
    plt.suptitle('Activation Atlas - Feature Inversions (Synthetic Optimal Inputs)', fontsize=16, y=1.001)
    plt.show()


def compare_real_vs_synthetic(model, dataloader, target_layer_index, layer_name, device='cpu', num_channels=4):
    """
    Show real images vs their synthetic inversions side-by-side.
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


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = AlexNet(num_classes=10).to(device)
    PATH = './alexnet_cifar10.pth'

    if os.path.exists(PATH):
        print("Loading trained model...")
        net.load_state_dict(torch.load(PATH, map_location=device))
        print(f"Model loaded from {PATH}")
    else:
        print("No pre-trained model found. Starting training...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        # the original AlexNet was trained over 90 epochs
        for epoch in range(90):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 100 == 99:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
        print('Finished Training')
        torch.save(net.state_dict(), PATH)
        print(f'Saved trained model to {PATH}')

    print("Generating embeddings for visualization...")
    embeddings, labels, images = get_embeddings(net, testloader, device)
    plot_embeddings(embeddings, labels, images, classes)

    print("\n" + "=" * 60)
    print("FEATURE PATTERN VISUALIZATION - TOP 64 ACTIVATED CHANNELS")
    print("=" * 60)

    # Configuration: Number of random sample images to visualize
    n_samples = 5  # Change this value to visualize more or fewer images

    # Get all test images for random sampling
    all_test_images = []
    all_test_labels = []
    for images_batch, labels_batch in testloader:
        all_test_images.append(images_batch)
        all_test_labels.append(labels_batch)
    all_test_images = torch.cat(all_test_images, dim=0)
    all_test_labels = torch.cat(all_test_labels, dim=0)

    # Randomly select n sample images
    total_images = len(all_test_images)
    random_indices = np.random.choice(total_images, size=min(n_samples, total_images), replace=False)

    conv_layer_indices = [i for i, layer in enumerate(net.features) if isinstance(layer, nn.Conv2d)]
    first_conv_layer = conv_layer_indices[0]  # First conv layer (outputs 64 channels)

    print(f"\nGenerating visualizations for {len(random_indices)} randomly selected images...")

    for idx, image_idx in enumerate(random_indices):
        sample_image = all_test_images[image_idx]
        sample_label = all_test_labels[image_idx].item()
        sample_label_name = classes[sample_label]

        print(f"\n[{idx + 1}/{len(random_indices)}] Processing image {image_idx} (Class: {sample_label_name})")

        # Generate Grad-CAM visualization
        print(f"  - Generating Grad-CAM visualization...")
        superimposed_img, heatmap = generate_grad_cam(net, sample_image)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        ax1.imshow(np.transpose(sample_image.numpy() / 2 + 0.5, (1, 2, 0)))
        ax1.set_title(f'Original Image: {sample_label_name}')
        ax1.axis('off')
        ax2.imshow(heatmap, cmap='jet')
        ax2.set_title('Grad-CAM Heatmap')
        ax2.axis('off')
        ax3.imshow(superimposed_img)
        ax3.set_title('Superimposed Image')
        ax3.axis('off')
        plt.tight_layout()
        # add some padding
        plt.subplots_adjust(wspace=0.3)
        plt.show()

        # Show top 64 activated feature maps
        print(f"  - Showing TOP 64 ACTIVATED FEATURE MAPS...")
        visualize_activation_maps_for_image(net, sample_image, target_layer_index=first_conv_layer,
                                            layer_name="Conv Layer 1",
                                            image_label=f"{sample_label_name} (#{image_idx})",
                                            num_maps=64)

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
