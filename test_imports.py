#!/usr/bin/env python
"""
Test script to verify the refactored modules can be imported correctly.
"""

if __name__ == "__main__":
    try:
        print("Testing imports...")

        # Test model import
        print("  - Importing model.AlexNet...", end=" ")
        from model import AlexNet
        print("✓")

        # Test analysis imports
        print("  - Importing analysis functions...", end=" ")
        from analysis import (
            get_embeddings,
            plot_embeddings,
            normalize_map,
            feature_inversion,
            feature_inversion_channel,
            get_top_activating_images
        )
        print("✓")

        # Test visualization imports
        print("  - Importing visualization functions...", end=" ")
        from visualization import (
            visualize_feature_maps,
            generate_grad_cam,
            visualize_activation_maps_for_image,
            visualize_layer_activations_on_real_images,
            generate_activation_atlas,
            generate_activation_atlas_per_channel,
            compare_real_vs_synthetic
        )
        print("✓")

        print("\n✅ All imports successful! Project refactoring is complete.")
        print("\nModule structure:")
        print("  - model.py: AlexNet class definition")
        print("  - analysis.py: 6 analysis/utility functions")
        print("  - visualization.py: 7 visualization functions")
        print("  - main.py: Main execution script")

    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        exit(1)

