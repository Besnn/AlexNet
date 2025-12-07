import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

from model import AlexNet
from analysis import get_embeddings, plot_embeddings
from visualization import (
    visualize_feature_maps,
    generate_grad_cam,
    visualize_activation_maps_for_image
)


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
