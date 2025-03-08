

"""
Image Model Initialization and Augmentation Visualization
--------------------------------------------------------
This script initializes MobileNetV2, InceptionV3, and ResNet50 models for inference.
It includes basic augmentation techniques to visualize their effect on uploaded images.
No command-line arguments needed - edit the IMAGE_PATH variable directly in the code.

Created on: 8/03/2025
Author: Miriam Asare- Baiden
"""

import os
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


# ===== EDIT THIS PATH TO YOUR IMAGE =====
IMAGE_PATH = "your image path goes here"  # Change this to your image path(could be a file or directory)

# Example: IMAGE_PATH = "/Users/username/Desktop/my_image.jpg"
# ========================================

# Number of images to display if IMAGE_PATH is a directory
DISPLAY_COUNT = 5


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Model Initialization and Image Augmentation')
    parser.add_argument('--image_path', type=str, required=True, help='Path to image or directory of images')
    parser.add_argument('--display_count', type=int, default=5, help='Number of images to display from a directory')
    
    return parser.parse_args()


def initialize_models():
    """Initialize pretrained models without modifying them for fine-tuning."""
    print("Initializing models...")
    
    # MobileNetV2
    mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    mobilenet.eval()
    print("✓ MobileNetV2 initialized")
    
    # InceptionV3
    inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    inception.eval()
    print("✓ InceptionV3 initialized")
    
    # ResNet50
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    resnet.eval()
    print("✓ ResNet50 initialized")
    
    return {
        'mobilenet': mobilenet,
        'inception': inception,
        'resnet': resnet
    }


def get_augmentation_techniques():
    """
    Define two key image augmentation techniques.
    Returns a dictionary of named transformations.
    """
    # Standard normalization for pretrained models
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Define augmentation techniques
    techniques = {
        'Original_image': transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            normalize
        ]),
        'Horizontal_flip': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            normalize
        ]),
        'Vertical_flip': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            normalize
        ])
    }
    
    return techniques
    
    return techniques


def load_image(image_path):
    """Load image from path and return PIL Image."""
    try:
        image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def apply_augmentations(image, augmentation_techniques):
    """Apply various augmentation techniques to an image."""
    augmented_images = {}
    
    for name, transform in augmentation_techniques.items():
        try:
            augmented = transform(image)
            
            # Convert for display
            if name == 'original':
                # Convert tensor to numpy for display
                img_np = augmented.permute(1, 2, 0).numpy()
                augmented_images[name] = img_np
            else:
                
                if isinstance(augmented, torch.Tensor):
                    
                    img_np = augmented.permute(1, 2, 0).numpy()
                    
                    # Reverse normalization approximately
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_np = img_np * std + mean
                    
                    img_np = np.clip(img_np, 0, 1)
                    augmented_images[name] = img_np
        except Exception as e:
            print(f"Error applying {name} augmentation: {e}")
    
    return augmented_images


def visualize_augmentations(augmented_images):
    """Visualize the original and augmented images."""
    n_images = len(augmented_images)
    n_cols = 3
    n_rows = 1
    
    plt.figure(figsize=(15, 5))
    
    for i, (name, img) in enumerate(augmented_images.items()):
        plt.subplot(n_rows, n_cols, i + 1)
        # All images are now numpy arrays with shape (H,W,C)
        plt.imshow(img)
        plt.title(name)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def save_augmented_images(image_path, augmented_images, output_dir):
    """Save each augmented image to disk."""
    image_name = Path(image_path).stem
    os.makedirs(output_dir, exist_ok=True)
    
    for name, img in augmented_images.items():
        if name == 'original':
            # Save the PIL Image
            output_path = os.path.join(output_dir, f"{image_name}_{name}.jpg")
            img.save(output_path)
        else:
            # Convert numpy array to PIL Image and save
            output_path = os.path.join(output_dir, f"{image_name}_{name}.jpg")
            plt.imsave(output_path, img)
    
    print(f"Saved all augmented versions of {image_name} to {output_dir}")


def process_image_or_directory(path, display_count):
    """Process a single image or a limited number of images from a directory."""
    augmentation_techniques = get_augmentation_techniques()
    
    # Check if path is a file or directory
    if os.path.isfile(path):
        # Process single image
        image = load_image(path)
        if image:
            print(f"Processing image: {path}")
            augmented_images = apply_augmentations(image, augmentation_techniques)
            visualize_augmentations(augmented_images)
    
    elif os.path.isdir(path):
        # Process a limited number of images in directory
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_paths = [os.path.join(path, f) for f in os.listdir(path) 
                      if os.path.isfile(os.path.join(path, f)) and 
                      f.lower().endswith(image_extensions)]
        
        # Limit the number of images to process
        image_paths = image_paths[:display_count]
        print(f"Found {len(image_paths)} images in directory {path}")
        
        for img_path in image_paths:
            image = load_image(img_path)
            if image:
                print(f"Processing image: {img_path}")
                augmented_images = apply_augmentations(image, augmentation_techniques)
                visualize_augmentations(augmented_images)
    
    else:
        print(f"Error: {path} is neither a file nor a directory")


def main():
    """Main function."""
    # Check CUDA availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize models
    models = initialize_models()
    
    # Process images
    process_image_or_directory(
        IMAGE_PATH,
        DISPLAY_COUNT
    )
    
    print("\nAll image processing completed!")
    print(f"\nModels initialized and ready for inference:")
    for model_name in models.keys():
        print(f"- {model_name}")


if __name__ == "__main__":
    main()