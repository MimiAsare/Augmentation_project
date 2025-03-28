"""
Erythema Classification System

This script provides a comprehensive framework for training and evaluating deep learning models
to classify erythema from different imaging modalities (optical, thermal black and white, 
and thermal color). It uses stratified k-fold cross-validation to ensure robust evaluation.

The system handles the complete pipeline including:
- Data loading and preprocessing
- Stratified group k-fold cross-validation setup
- Model training with early stopping
- Comprehensive evaluation metrics
- Visualization of results

Main components:
- setup_paths: Configures file paths for the dataset
- get_valid_images: Loads valid images and labels from directories
- stratified_kfold: Creates stratified cross-validation folds
- model_train_evaluation: Trains and evaluates models across folds
- run_pipeline: Orchestrates the entire workflow

Usage:
1. Modify the CONFIG dictionary with your file paths and desired settings
2. Run the script

Example:
    python erythema_classification.py

Authors: Miriam Asare- Baiden
Date: March 2025
"""

# Import necessary libraries
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights, inception_v3, Inception_V3_Weights, resnet50, ResNet50_Weights
from PIL import Image
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix as cm, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

# Configuration - MODIFY THESE VALUES AS NEEDED
CONFIG = {
    # File paths - replace with your actual paths
    'excel_path': "/Users/username/path/to/ErythemaImgLabels.xlsx",
    'erythema_path': "/Users/username/path/to/ProcessedData/Separated & Processed Images/",
    'monk_filepath': "/Users/username/path/to/Colorimeter-Demographics_Combined.xlsx",
    'save_dir': "./Saved_models",
    
    # Model configuration
    'which_model': "MobileNetV2",  # Options: "MobileNetV2", "InceptionNetV3", "ResNet50"
    'img_size': (224, 224),  # Use (299, 299) for InceptionNetV3
    
    # Training parameters
    'batch_size': 32,
    'num_epochs': 100,
    'patience': 15,
    'seed': 42
}


def setup_paths(config):
    """
    Setup the paths for data access.
    
    This function takes the configuration dictionary and creates the specific paths
    for each image modality (optical, thermal black and white, thermal color).
    It also ensures the save directory exists.
    
    Args:
        config (dict): Dictionary containing configuration settings including file paths
        
    Returns:
        tuple: Tuple containing (erythema_optical_path, erythema_thermal_bw_path, erythema_thermal_color_path)
    """
    # Ensure erythema_path ends with a slash
    erythema_path = config['erythema_path']
    if not erythema_path.endswith('/'):
        erythema_path += '/'
    
    # Define the specific paths for different image types
    erythema_optical_path = f'{erythema_path}only_cupping_images_optical'
    erythema_thermal_bw_path = f'{erythema_path}only_cupping_images'
    erythema_thermal_color_path = f'{erythema_path}only_cupping_images_color'
    
    # Create save directory if it doesn't exist
    os.makedirs(config['save_dir'], exist_ok=True)
    
    return erythema_optical_path, erythema_thermal_bw_path, erythema_thermal_color_path

def get_valid_images(image_dir, erythema_file):
    """
    Load valid images and their labels from directory.
    
    This function takes a directory path and an Excel file with image labels,
    validates that each image exists, and returns lists of valid image paths
    and their corresponding labels.
    
    Args:
        image_dir (str): Directory containing the images
        erythema_file (pandas.DataFrame): DataFrame containing image names and labels
        
    Returns:
        tuple: Lists of (train_images, train_labels) containing valid image paths and labels
    """
    train_images = []
    train_labels = []
   
    image_names = erythema_file['Img Name'].values
    labels = erythema_file['Label'].values
    for img_name, label in zip(image_names, labels):
        img_name = img_name.strip()  # Remove any leading/trailing whitespace
        image_path = os.path.join(image_dir, img_name)
      
        if os.path.exists(image_path):
            train_images.append(image_path)
            train_labels.append(label)
        else:
            print(f"Warning: Image not found: {image_path}")
    print(f'Total number of train images and train labels are:', len(train_images), len(train_labels))

    return train_images, train_labels


def get_fold_train_test(data, labels, test_subjects):
    """
    Split data into train and test sets based on test_subjects.
    
    This function takes image paths, labels, and a list of test subject IDs,
    and splits the data into training and testing sets based on subject ID.
    
    Args:
        data (list): List of image file paths
        labels (list): List of labels corresponding to each image
        test_subjects (list): List of subject IDs to include in test set
        
    Returns:
        tuple: (train_fold, train_fold_labels, test_fold, test_fold_labels)
    """
    # initialize list for optical train and test folds
    train_fold = []
    test_fold = []
    train_fold_labels = []
    test_fold_labels = []

    data_label_dict = dict(zip(data, labels))

    for file in data:
        file_parts = file.split('/')[-1]
        subj_id = '_'.join(file_parts.split('_')[:2])
       
        if subj_id not in test_subjects:
            train_fold.append(file)
            train_fold_labels.append(data_label_dict[file])
        else:
            test_fold.append(file)
            test_fold_labels.append(data_label_dict[file])
    
    return train_fold, train_fold_labels, test_fold, test_fold_labels

def stratified_kfold(erythema_optical_path, erythema_thermal_bw_path, erythema_thermal_color_path, monk_filepath, erythema_file):
    """
    Perform stratified k-fold cross validation.
    
    This function creates stratified k-fold cross-validation splits based on Monk skin tone groups,
    ensuring that subjects with similar skin tones are distributed across folds.
    
    Args:
        erythema_optical_path (str): Path to optical images
        erythema_thermal_bw_path (str): Path to thermal black and white images
        erythema_thermal_color_path (str): Path to thermal color images
        monk_filepath (str): Path to Excel file with Monk skin tone groups
        erythema_file (pandas.DataFrame): DataFrame with image labels
        
    Returns:
        tuple: (optical_ery_folds, thermal_bw_ery_folds, thermal_color_ery_folds)
               Each is a list of dictionaries with 'train_images', 'train_labels',
               'test_images', and 'test_labels' for each fold
    """
    optical_ery_folds = []
    thermal_bw_ery_folds = []
    thermal_color_ery_folds = []

    # Get the images and labels per modality
    optical_ery, optical_ery_labels = get_valid_images(erythema_optical_path, erythema_file)
    thermal_bw_ery, thermal_bw_ery_labels = get_valid_images(erythema_thermal_bw_path, erythema_file)
    thermal_color_ery, thermal_color_ery_labels = get_valid_images(erythema_thermal_color_path, erythema_file)

    # Get stratified k-fold using the subject id for grouping
    sub_monk_skin_tone = pd.read_excel(monk_filepath).iloc[0:35, :]

    subject_ids = sub_monk_skin_tone['Subj_ID']
    monk_scores = sub_monk_skin_tone['Monk_Group']

    # Initialize StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CONFIG['seed'])

    for fold_idx, (_, test_idx) in enumerate(sgkf.split(subject_ids, monk_scores, groups=subject_ids)):
        test_subjects = subject_ids.iloc[test_idx].tolist()
        print(f"Test subjects for fold {fold_idx+1}:", test_subjects)
       
        train_fold_optical_ery, train_fold_optical_ery_labels, test_fold_optical_ery, test_fold_optical_ery_labels = get_fold_train_test(optical_ery, optical_ery_labels, test_subjects)
        train_fold_thermal_bw_ery, train_fold_thermal_bw_ery_labels, test_fold_thermal_bw_ery, test_fold_thermal_bw_ery_labels = get_fold_train_test(thermal_bw_ery, thermal_bw_ery_labels, test_subjects)
        train_fold_thermal_color_ery, train_fold_thermal_color_ery_labels, test_fold_thermal_color_ery, test_fold_thermal_color_ery_labels = get_fold_train_test(thermal_color_ery, thermal_color_ery_labels, test_subjects)
        
        optical_ery_folds.append({
            'train_images': train_fold_optical_ery, 
            'train_labels': train_fold_optical_ery_labels, 
            'test_images': test_fold_optical_ery, 
            'test_labels': test_fold_optical_ery_labels
        })
        
        thermal_bw_ery_folds.append({
            'train_images': train_fold_thermal_bw_ery, 
            'train_labels': train_fold_thermal_bw_ery_labels,
            'test_images': test_fold_thermal_bw_ery, 
            'test_labels': test_fold_thermal_bw_ery_labels
        })
        
        thermal_color_ery_folds.append({
            'train_images': train_fold_thermal_color_ery, 
            'train_labels': train_fold_thermal_color_ery_labels,
            'test_images': test_fold_thermal_color_ery, 
            'test_labels': test_fold_thermal_color_ery_labels
        })
        
        print(f'Fold {fold_idx+1} extracted')
    return optical_ery_folds, thermal_bw_ery_folds, thermal_color_ery_folds

class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset for loading erythema images.
    
    This class extends PyTorch's Dataset class to load and transform images
    for training and evaluation. It handles label mapping to ensure consistent
    numeric label assignments.
    
    Args:
        image_paths (list): List of image file paths
        labels (list): List of labels corresponding to each image
        transform (callable, optional): Optional transform to be applied to images
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.label_map = {label: idx for idx, label in enumerate(set(labels))}

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (image, label) where image is a transformed PIL Image and
                   label is the corresponding class index
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        label_idx = self.label_map[label]
        return image, torch.tensor(label_idx, dtype=torch.long)

def visualize_augmentations(dataset, image_size, num_samples=3, img_indices=None):
    """
    Visualize augmentations applied to training images.
    
    This function displays the original images alongside various augmentations
    to help understand how data augmentation is affecting the training data.
    
    Args:
        dataset (CustomDataset): The dataset containing images to visualize
        image_size (tuple): Size (width, height) for resizing images
        num_samples (int, optional): Number of random samples to visualize. Defaults to 3.
        img_indices (list, optional): Specific image indices to visualize. 
                                     If None, random indices are chosen. Defaults to None.
    
    Returns:
        matplotlib.figure.Figure: The created figure object for further customization if needed
    """
    # If no specific indices provided, select random ones
    if img_indices is None:
        img_indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    # Original transform components (for individual augmentations)
    resize = transforms.Resize(image_size)
    rotate = transforms.RandomRotation(20)
    h_flip = transforms.RandomHorizontalFlip(p=1.0)  # Always flip
    v_flip = transforms.RandomVerticalFlip(p=1.0)    # Always flip
    
    figures = []
    
    # Create a figure with subplots
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    # Process each selected image
    for i, idx in enumerate(img_indices):
        # Get the image path and label
        image_path = dataset.image_paths[idx]
        label = dataset.labels[idx]
        
        # Load original image
        original_img = Image.open(image_path).convert('RGB')
        
        # Apply individual transformations
        resized_img = resize(original_img)
        rotated_img = rotate(resized_img.copy())
        h_flipped_img = h_flip(resized_img.copy())
        v_flipped_img = v_flip(resized_img.copy())
        
        # Display images
        axes[i, 0].imshow(resized_img)
        axes[i, 0].set_title(f'Original Image')
        
        axes[i, 1].imshow(rotated_img)
        axes[i, 1].set_title('Rotation at 20 degrees')
        
        axes[i, 2].imshow(h_flipped_img)
        axes[i, 2].set_title('Horizontal Flip')
        
        axes[i, 3].imshow(v_flipped_img)
        axes[i, 3].set_title('Vertical Flip')
        
        # Remove axis ticks
        for j in range(4):
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def initialize_model(which_model, num_classes):
    """
    Initialize model architecture based on user choice.
    
    This function creates and configures a deep learning model according to the specified 
    architecture, modifying the final layer to match the number of classes in the dataset.
    
    Args:
        which_model (str): Model architecture to use ('MobileNetV2', 'InceptionNetV3', or 'ResNet50')
        num_classes (int): Number of output classes
        
    Returns:
        torch.nn.Module: Initialized model with pre-trained weights and modified final layer
        
    Raises:
        ValueError: If an unsupported model type is specified
    """
    if which_model == 'MobileNetV2':
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif which_model == 'InceptionNetV3':
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.aux_logits = True  # Enable auxiliary logits
        return model
    elif which_model == 'ResNet50':
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unsupported model: {which_model}")

def model_train_evaluation(data_in_folds, name, which_model, img_size, batch_size, num_epochs, patience, save_dir):
    """
    Train and evaluate model using k-fold cross validation.
    
    This is the core function that handles the complete training and evaluation pipeline:
    - Sets up data loaders for each fold
    - Initializes the model
    - Trains the model with early stopping
    - Evaluates performance metrics
    - Creates visualizations
    - Saves model checkpoints
    
    Args:
        data_in_folds (list): List of dictionaries containing training and test data for each fold
        name (str): Name of the modality ('optical', 'thermal_bw', 'thermal_color')
        which_model (str): Model architecture to use
        img_size (tuple): Image size as (width, height)
        batch_size (int): Batch size for training
        num_epochs (int): Maximum number of epochs for training
        patience (int): Early stopping patience (epochs with no improvement)
        save_dir (str): Directory to save model checkpoints and results
        
    Returns:
        dict: Dictionary of metrics for all folds including accuracy, AUC, etc.
    """
    print(f'Image size is {img_size}')

    # Set seeds for reproducibility
    torch.manual_seed(CONFIG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CONFIG['seed'])
        torch.cuda.manual_seed_all(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    random.seed(CONFIG['seed'])

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Metrics storage
    model_accuracy = []
    model_auc = []
    model_specificity = []
    model_sensitivity = []
    model_f1 = []
    all_true_labels_across_folds = []
    all_pred_labels_across_folds = []
    total_correctly_classified = []
    total_correct_labels = []
    total_correct_preds = []
    total_misclassified = []
    total_misclass_labels = []
    total_misclass_pred = []

    # Create transforms
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Run model on each fold
    print(f"Model training and evaluation on {name} modality")

    for fold_idx, fold in enumerate(data_in_folds):
        print("="*50)
        print(f'Starting Fold {fold_idx+1}')
        print("="*50)

        fold_train_paths = fold['train_images']
        fold_train_labels = fold['train_labels']
        fold_val_paths = fold['test_images']
        fold_val_labels = fold['test_labels']

        # Print the total training and evaluation images
        print(f'Total training images and labels are:', len(fold_train_paths), len(fold_train_labels))
        print(f'Total evaluation images and labels are:', len(fold_val_paths), len(fold_val_labels))
    
        train_dataset = CustomDataset(fold_train_paths, fold_train_labels, transform=train_transform)
        
        # Visualize some augmented images (only for first fold)
        if fold_idx == 0:
            visualize_augmentations(train_dataset, img_size, num_samples=3)
        
        val_dataset = CustomDataset(fold_val_paths, fold_val_labels, transform=val_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Initialize the model
        model = initialize_model(which_model, len(set(fold_train_labels)))
        model.to(device)

        # Define criterion and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        best_val_auc = 0.0
        epochs_no_improve = 0
        best_model_state = None
        best_val_metrics = None

        # Training Loop
        print('Starting training loop')
        for epoch in range(1, num_epochs + 1):
            model.train()
            running_loss = 0.0
            for data in train_loader:
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                
                # Handle InceptionV3 training phase
                if which_model == 'InceptionNetV3' and model.training:
                    outputs, aux_outputs = model(inputs)
                    loss1 = criterion(outputs, targets)
                    loss2 = criterion(aux_outputs, targets)
                    loss = loss1 + 0.4 * loss2
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f'Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}')
            
            # Validation
            model.eval()
            val_corrects = 0
            val_total = 0
            val_all_labels = []
            val_all_probs = []
            val_all_preds = []

            correctly_classified = []
            correct_labels = []
            correct_preds = []
            misclassified = []
            misclass_labels = []
            misclass_preds = []

            with torch.no_grad():
                for i, (inputs, targets) in enumerate(val_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # Handle InceptionV3 evaluation phase
                    if which_model == 'InceptionNetV3':
                        outputs = model(inputs)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                    else:
                        outputs = model(inputs)
                        
                    _, preds = torch.max(outputs, 1)
                    
                    # Track image paths for analyzing errors
                    start_idx = i * val_loader.batch_size
                    for j in range(len(preds)):
                        idx = start_idx + j
                        if idx < len(fold_val_paths):  # Avoid index out of range
                            image_path = fold_val_paths[idx]
                            label = targets[j].item()
                            pred = preds[j].item()
                            if pred == label:
                                correctly_classified.append(image_path)
                                correct_labels.append(label)
                                correct_preds.append(pred)
                            else:
                                misclassified.append(image_path)
                                misclass_labels.append(label)
                                misclass_preds.append(pred)
                    
                    # Update metrics
                    val_total += targets.size(0)
                    val_corrects += (preds == targets).sum().item()
                    val_all_labels.extend(targets.cpu().numpy())
                    
                    # Get probabilities for AUC calculation
                    probs = torch.softmax(outputs, dim=1)
                    if probs.shape[1] == 2:  # Binary case
                        val_all_probs.extend(probs[:, 1].cpu().numpy())
                    else:  # Multi-class case
                        val_all_probs.extend(probs.cpu().numpy())
                    
                    val_all_preds.extend(preds.cpu().numpy())

            # Compute metrics
            val_acc = val_corrects / val_total
            
            # Calculate AUC (handle multi-class case)
            if len(set(val_all_labels)) == 2:
                val_auc = roc_auc_score(val_all_labels, val_all_probs)
                # Binary metrics
                sensitivity = recall_score(val_all_labels, val_all_preds)
                tn, fp, fn, tp = cm(val_all_labels, val_all_preds).ravel()
                specificity = tn / (tn + fp)
                f1 = f1_score(val_all_labels, val_all_preds)
            else:
                # Multi-class metrics (using macro averaging)
                val_auc = roc_auc_score(val_all_labels, val_all_probs, multi_class='ovr')
                sensitivity = recall_score(val_all_labels, val_all_preds, average='macro')
                f1 = f1_score(val_all_labels, val_all_preds, average='macro')
                # No simple specificity for multi-class
                specificity = 0.0

            print(
                f"Fold {fold_idx + 1} - Validation Accuracy: {val_acc:.3f}, "
                f"Sensitivity: {sensitivity:.3f}, "
                f"Specificity: {specificity:.3f}, "
                f"AUC: {val_auc:.3f}, "
                f"F1: {f1:.3f}"
            )

            # Check and update the best model for this fold
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                epochs_no_improve = 0
                best_model_state = model.state_dict()
                best_val_metrics = {
                    'val_acc': val_acc,
                    'val_auc': val_auc,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'f1': f1
                }
                temp_correctly_classified = correctly_classified
                temp_correct_labels = correct_labels
                temp_correct_preds = correct_preds
                temp_misclassified = misclassified
                temp_misclass_labels = misclass_labels
                temp_misclass_pred = misclass_preds
                temp_val_all_labels = val_all_labels
                temp_val_all_preds = val_all_preds
            else:
                epochs_no_improve += 1
                print(f'No performance gains at: {epochs_no_improve}')
            
            # Early stopping
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch} for fold {fold_idx + 1}')
                break

        # Update metrics for this fold
        model_accuracy.append(best_val_metrics['val_acc'])
        model_auc.append(best_val_metrics['val_auc'])
        model_specificity.append(best_val_metrics['specificity'])
        model_sensitivity.append(best_val_metrics['sensitivity'])
        model_f1.append(best_val_metrics['f1'])

        # Collect results across folds
        total_correctly_classified.extend(temp_correctly_classified)
        total_misclassified.extend(temp_misclassified)
        total_correct_preds.extend(temp_correct_preds)
        total_misclass_pred.extend(temp_misclass_pred)
        total_correct_labels.extend(temp_correct_labels)
        total_misclass_labels.extend(temp_misclass_labels)
        all_true_labels_across_folds.extend(temp_val_all_labels)
        all_pred_labels_across_folds.extend(temp_val_all_preds)
        
        # Save best model for this fold
        fold_save_path = os.path.join(save_dir, f'{name}_{which_model}_fold{fold_idx+1}.pth')
        torch.save(best_model_state, fold_save_path)
        print(f'Fold {fold_idx+1} model saved to {fold_save_path}')

    # Save the best model's state dictionary across all folds
    save_model_path = os.path.join(save_dir, f'{name}_{which_model}_stratified_gkf_model.pth')
    torch.save(best_model_state, save_model_path)
    print(f'Best model saved to {save_model_path}')

    # Get the mean and standard deviation of evaluation metrics
    model_accuracy_mean = np.mean(model_accuracy)
    model_accuracy_std = np.std(model_accuracy)
    model_auc_mean = np.mean(model_auc)
    model_auc_std = np.std(model_auc)
    model_specificity_mean = np.mean(model_specificity)
    model_specificity_std = np.std(model_specificity)
    model_sensitivity_mean = np.mean(model_sensitivity)
    model_sensitivity_std = np.std(model_sensitivity)
    model_f1_mean = np.mean(model_f1)
    model_f1_std = np.std(model_f1)

    # Print metrics: mean and std
    print(f'Mean and standard deviation of evaluation metrics for {which_model} {name}:')
    print(f'Accuracy: {model_accuracy_mean:.3f}± {model_accuracy_std:.3f}')
    print(f'AUC: {model_auc_mean:.3f} ± {model_auc_std:.3f}')
    print(f'Sensitivity: {model_sensitivity_mean:.3f} ± {model_sensitivity_std:.3f}')
    print(f'Specificity: {model_specificity_mean:.3f} ± {model_specificity_std:.3f}')
    print(f'F1: {model_f1_mean:.3f} ± {model_f1_std:.3f}')

    # Show the list of top aucs per fold
    print(f'Best AUCs across all folds for {name}: {model_auc}')
    
    # Create confusion matrix display
    if len(set(all_true_labels_across_folds)) == 2:
        display_labels = ['Erythema', 'Base']
    else:
        display_labels = [str(i) for i in range(len(set(all_true_labels_across_folds)))]
    
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay.from_predictions(
        all_true_labels_across_folds, 
        all_pred_labels_across_folds, 
        display_labels=display_labels,
        cmap='YlGnBu'
    )

    # Access the axis object and set font sizes
    ax = disp.ax_
    plt.setp(ax.get_yticklabels(), fontsize=14)  # For y-axis labels
    plt.setp(ax.get_xticklabels(), fontsize=14)  # For x-axis labels
    # modify the fontsize of the x and y axes
    ax.set_xlabel(ax.get_xlabel(), fontsize=15)
    ax.set_ylabel(ax.get_ylabel(), fontsize=15)

    # To adjust the numbers inside the cells
    for im in ax.images:
        im.colorbar.ax.tick_params(labelsize=16)  # For colorbar text
        
    # Adjust only the numbers inside the cells
    for text in disp.ax_.texts:
        text.set_fontsize(18)  

    plt.title(f'{name} modality - {which_model}')
    plt.tight_layout()
    confusion_matrix_path = os.path.join(save_dir, f'{name}_{which_model}_confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    plt.show()
    
    # Return metrics for further analysis
    metrics = {
        'accuracy': model_accuracy,
        'auc': model_auc,
        'specificity': model_specificity,
        'sensitivity': model_sensitivity,
        'f1': model_f1,
        'correctly_classified': total_correctly_classified,
        'misclassified': total_misclassified
    }
    
    # return metricsavailable() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the model
    model = initialize_model(which_model, len(set(fold_train_labels)))
    model.to(device)

    # Define criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_auc = 0.0
    epochs_no_improve = 0
    best_model_state = None
    best_val_metrics = None

    # Training Loop
    print('Starting training loop')
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Handle InceptionV3 training phase
            if which_model == 'InceptionNetV3' and model.training:
                outputs, aux_outputs = model(inputs)
                loss1 = criterion(outputs, targets)
                loss2 = criterion(aux_outputs, targets)
                loss = loss1 + 0.4 * loss2
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}')
        
        # Validation
        model.eval()
        val_corrects = 0
        val_total = 0
        val_all_labels = []
        val_all_probs = []
        val_all_preds = []

        correctly_classified = []
        correct_labels = []
        correct_preds = []
        misclassified = []
        misclass_labels = []
        misclass_preds = []

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Handle InceptionV3 evaluation phase
                if which_model == 'InceptionNetV3':
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                else:
                    outputs = model(inputs)
                    
                _, preds = torch.max(outputs, 1)
                
                # Track image paths for analyzing errors
                start_idx = i * val_loader.batch_size
                for j in range(len(preds)):
                    idx = start_idx + j
                    if idx < len(fold_val_paths):  # Avoid index out of range
                        image_path = fold_val_paths[idx]
                        label = targets[j].item()
                        pred = preds[j].item()
                        if pred == label:
                            correctly_classified.append(image_path)
                            correct_labels.append(label)
                            correct_preds.append(pred)
                        else:
                            misclassified.append(image_path)
                            misclass_labels.append(label)
                            misclass_preds.append(pred)
                
                # Update metrics
                val_total += targets.size(0)
                val_corrects += (preds == targets).sum().item()
                val_all_labels.extend(targets.cpu().numpy())
                
                # Get probabilities for AUC calculation
                probs = torch.softmax(outputs, dim=1)
                if probs.shape[1] == 2:  # Binary case
                    val_all_probs.extend(probs[:, 1].cpu().numpy())
                else:  # Multi-class case
                    val_all_probs.extend(probs.cpu().numpy())
                
                val_all_preds.extend(preds.cpu().numpy())

        # Compute metrics
        val_acc = val_corrects / val_total
        
        # Calculate AUC (handle multi-class case)
        if len(set(val_all_labels)) == 2:
            val_auc = roc_auc_score(val_all_labels, val_all_probs)
            # Binary metrics
            sensitivity = recall_score(val_all_labels, val_all_preds)
            tn, fp, fn, tp = cm(val_all_labels, val_all_preds).ravel()
            specificity = tn / (tn + fp)
            f1 = f1_score(val_all_labels, val_all_preds)
        else:
            # Multi-class metrics (using macro averaging)
            val_auc = roc_auc_score(val_all_labels, val_all_probs, multi_class='ovr')
            sensitivity = recall_score(val_all_labels, val_all_preds, average='macro')
            f1 = f1_score(val_all_labels, val_all_preds, average='macro')
            # No simple specificity for multi-class
            specificity = 0.0

        print(
            f"Fold {fold_idx + 1} - Validation Accuracy: {val_acc:.3f}, "
            f"Sensitivity: {sensitivity:.3f}, "
            f"Specificity: {specificity:.3f}, "
            f"AUC: {val_auc:.3f}, "
            f"F1: {f1:.3f}"
        )

        # Check and update the best model for this fold
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            best_val_metrics = {
                'val_acc': val_acc,
                'val_auc': val_auc,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'f1': f1
            }
            temp_correctly_classified = correctly_classified
            temp_correct_labels = correct_labels
            temp_correct_preds = correct_preds
            temp_misclassified = misclassified
            temp_misclass_labels = misclass_labels
            temp_misclass_pred = misclass_preds
            temp_val_all_labels = val_all_labels
            temp_val_all_preds = val_all_preds
        else:
            epochs_no_improve += 1
            print(f'No performance gains at: {epochs_no_improve}')
        
        # Early stopping
        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch} for fold {fold_idx + 1}')
            break

    # Update metrics for this fold
    model_accuracy.append(best_val_metrics['val_acc'])
    model_auc.append(best_val_metrics['val_auc'])
    model_specificity.append(best_val_metrics['specificity'])
    model_sensitivity.append(best_val_metrics['sensitivity'])
    model_f1.append(best_val_metrics['f1'])

    # Collect results across folds
    total_correctly_classified.extend(temp_correctly_classified)
    total_misclassified.extend(temp_misclassified)
    total_correct_preds.extend(temp_correct_preds)
    total_misclass_pred.extend(temp_misclass_pred)
    total_correct_labels.extend(temp_correct_labels)
    total_misclass_labels.extend(temp_misclass_labels)
    all_true_labels_across_folds.extend(temp_val_all_labels)
    all_pred_labels_across_folds.extend(temp_val_all_preds)
    
    # Save best model for this fold
    fold_save_path = os.path.join(save_dir, f'{name}_{which_model}_fold{fold_idx+1}.pth')
    torch.save(best_model_state, fold_save_path)
    print(f'Fold {fold_idx+1} model saved to {fold_save_path}')

    # Save the best model's state dictionary across all folds
    save_model_path = os.path.join(save_dir, f'{name}_{which_model}_stratified_gkf_model.pth')
    torch.save(best_model_state, save_model_path)
    print(f'Best model saved to {save_model_path}')

    # Get the mean and standard deviation of evaluation metrics
    model_accuracy_mean = np.mean(model_accuracy)
    model_accuracy_std = np.std(model_accuracy)
    model_auc_mean = np.mean(model_auc)
    model_auc_std = np.std(model_auc)
    model_specificity_mean = np.mean(model_specificity)
    model_specificity_std = np.std(model_specificity)
    model_sensitivity_mean = np.mean(model_sensitivity)
    model_sensitivity_std = np.std(model_sensitivity)
    model_f1_mean = np.mean(model_f1)
    model_f1_std = np.std(model_f1)

    # Print metrics: mean and std
    print(f'Mean and standard deviation of evaluation metrics for {which_model} {name}:')
    print(f'Accuracy: {model_accuracy_mean:.3f}± {model_accuracy_std:.3f}')
    print(f'AUC: {model_auc_mean:.3f} ± {model_auc_std:.3f}')
    print(f'Sensitivity: {model_sensitivity_mean:.3f} ± {model_sensitivity_std:.3f}')
    print(f'Specificity: {model_specificity_mean:.3f} ± {model_specificity_std:.3f}')
    print(f'F1: {model_f1_mean:.3f} ± {model_f1_std:.3f}')

    # Show the list of top aucs per fold
    print(f'Best AUCs across all folds for {name}: {model_auc}')
    
    # Create confusion matrix display
    if len(set(all_true_labels_across_folds)) == 2:
        display_labels = ['Erythema', 'Base']
    else:
        display_labels = [str(i) for i in range(len(set(all_true_labels_across_folds)))]
    
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay.from_predictions(
        all_true_labels_across_folds, 
        all_pred_labels_across_folds, 
        display_labels=display_labels,
        cmap='YlGnBu'
    )

    # Access the axis object and set font sizes
    ax = disp.ax_
    plt.setp(ax.get_yticklabels(), fontsize=14)  # For y-axis labels
    plt.setp(ax.get_xticklabels(), fontsize=14)  # For x-axis labels
    # modify the fontsize of the x and y axes
    ax.set_xlabel(ax.get_xlabel(), fontsize=15)
    ax.set_ylabel(ax.get_ylabel(), fontsize=15)

    # To adjust the numbers inside the cells
    for im in ax.images:
        im.colorbar.ax.tick_params(labelsize=16)  # For colorbar text
        
    # Adjust only the numbers inside the cells
    for text in disp.ax_.texts:
        text.set_fontsize(18)  

    plt.title(f'{name} modality - {which_model}')
    plt.tight_layout()
    confusion_matrix_path = os.path.join(save_dir, f'{name}_{which_model}_confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    plt.show()
    
    # Return metrics for further analysis
    metrics = {
        'accuracy': model_accuracy,
        'auc': model_auc,
        'specificity': model_specificity,
        'sensitivity': model_sensitivity,
        'f1': model_f1,
        'correctly_classified': total_correctly_classified,
        'misclassified': total_misclassified
    }
    
    return metrics

# Main function to run the entire pipeline
def run_pipeline():
    """
    Run the complete pipeline for training and evaluation.
    
    This is the main entry point for the image classification system.
    It orchestrates the entire workflow:
    1. Setting up paths
    2. Loading data
    3. Creating stratified k-fold splits
    4. Training and evaluating models on each modality
    5. Comparing results across modalities
    6. Generating visualizations
    7. Saving results
    
    Returns:
        dict: Dictionary containing metrics for each modality
    """
    # Setup paths
    erythema_optical_path, erythema_thermal_bw_path, erythema_thermal_color_path = setup_paths(CONFIG)
    
    # Load data
    erythema_file = pd.read_excel(CONFIG['excel_path'])
    
    # Perform stratified k-fold
    optical_ery_folds, thermal_bw_ery_folds, thermal_color_ery_folds = stratified_kfold(
        erythema_optical_path, 
        erythema_thermal_bw_path, 
        erythema_thermal_color_path, 
        CONFIG['monk_filepath'],
        erythema_file
    )
    
    # Initialize the selected model
    img_size = CONFIG['img_size']
    
    # Train and evaluate on each modality
    print("\nTraining and evaluating on Optical modality...")
    optical_metrics = model_train_evaluation(
        optical_ery_folds, 
        "optical", 
        CONFIG['which_model'], 
        img_size, 
        CONFIG['batch_size'], 
        CONFIG['num_epochs'], 
        CONFIG['patience'],
        CONFIG['save_dir']
    )
    
    print("\nTraining and evaluating on Thermal BW modality...")
    thermal_bw_metrics = model_train_evaluation(
        thermal_bw_ery_folds, 
        "thermal_bw", 
        CONFIG['which_model'], 
        img_size, 
        CONFIG['batch_size'], 
        CONFIG['num_epochs'], 
        CONFIG['patience'],
        CONFIG['save_dir']
    )
    
    print("\nTraining and evaluating on Thermal Color modality...")
    thermal_color_metrics = model_train_evaluation(
        thermal_color_ery_folds, 
        "thermal_color", 
        CONFIG['which_model'], 
        img_size, 
        CONFIG['batch_size'], 
        CONFIG['num_epochs'], 
        CONFIG['patience'],
        CONFIG['save_dir']
    )
    
    # Compare performance across modalities
    print("\nComparison of performance across modalities:")
    print(f"{'Modality':<15} | {'Accuracy':<20} | {'AUC':<20} | {'Sensitivity':<20} | {'Specificity':<20} | {'F1':<20}")
    print("-" * 115)
    
    optical_means = f"{np.mean(optical_metrics['accuracy']):.3f}±{np.std(optical_metrics['accuracy']):.3f}"
    optical_auc = f"{np.mean(optical_metrics['auc']):.3f}±{np.std(optical_metrics['auc']):.3f}"
    optical_sens = f"{np.mean(optical_metrics['sensitivity']):.3f}±{np.std(optical_metrics['sensitivity']):.3f}"
    optical_spec = f"{np.mean(optical_metrics['specificity']):.3f}±{np.std(optical_metrics['specificity']):.3f}"
    optical_f1 = f"{np.mean(optical_metrics['f1']):.3f}±{np.std(optical_metrics['f1']):.3f}"
    
    thermal_bw_means = f"{np.mean(thermal_bw_metrics['accuracy']):.3f}±{np.std(thermal_bw_metrics['accuracy']):.3f}"
    thermal_bw_auc = f"{np.mean(thermal_bw_metrics['auc']):.3f}±{np.std(thermal_bw_metrics['auc']):.3f}"
    thermal_bw_sens = f"{np.mean(thermal_bw_metrics['sensitivity']):.3f}±{np.std(thermal_bw_metrics['sensitivity']):.3f}"
    thermal_bw_spec = f"{np.mean(thermal_bw_metrics['specificity']):.3f}±{np.std(thermal_bw_metrics['specificity']):.3f}"
    thermal_bw_f1 = f"{np.mean(thermal_bw_metrics['f1']):.3f}±{np.std(thermal_bw_metrics['f1']):.3f}"
    
    thermal_color_means = f"{np.mean(thermal_color_metrics['accuracy']):.3f}±{np.std(thermal_color_metrics['accuracy']):.3f}"
    thermal_color_auc = f"{np.mean(thermal_color_metrics['auc']):.3f}±{np.std(thermal_color_metrics['auc']):.3f}"
    thermal_color_sens = f"{np.mean(thermal_color_metrics['sensitivity']):.3f}±{np.std(thermal_color_metrics['sensitivity']):.3f}"
    thermal_color_spec = f"{np.mean(thermal_color_metrics['specificity']):.3f}±{np.std(thermal_color_metrics['specificity']):.3f}"
    thermal_color_f1 = f"{np.mean(thermal_color_metrics['f1']):.3f}±{np.std(thermal_color_metrics['f1']):.3f}"
    
    print(f"{'Optical':<15} | {optical_means:<20} | {optical_auc:<20} | {optical_sens:<20} | {optical_spec:<20} | {optical_f1:<20}")
    print(f"{'Thermal BW':<15} | {thermal_bw_means:<20} | {thermal_bw_auc:<20} | {thermal_bw_sens:<20} | {thermal_bw_spec:<20} | {thermal_bw_f1:<20}")
    print(f"{'Thermal Color':<15} | {thermal_color_means:<20} | {thermal_color_auc:<20} | {thermal_color_sens:<20} | {thermal_color_spec:<20} | {thermal_color_f1:<20}")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Modality': ['Optical', 'Thermal BW', 'Thermal Color'],
        'Accuracy': [optical_means, thermal_bw_means, thermal_color_means],
        'AUC': [optical_auc, thermal_bw_auc, thermal_color_auc],
        'Sensitivity': [optical_sens, thermal_bw_sens, thermal_color_sens],
        'Specificity': [optical_spec, thermal_bw_spec, thermal_color_spec],
        'F1': [optical_f1, thermal_bw_f1, thermal_color_f1]
    })
    
    results_path = os.path.join(CONFIG['save_dir'], f"{CONFIG['which_model']}_results_comparison.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    
    # Create bar chart comparing metrics across modalities
    metrics_to_plot = ['accuracy', 'auc', 'sensitivity', 'specificity', 'f1']
    modalities = ['Optical', 'Thermal BW', 'Thermal Color']
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(20, 6))
    
    for i, metric in enumerate(metrics_to_plot):
        means = [
            np.mean(optical_metrics[metric]),
            np.mean(thermal_bw_metrics[metric]),
            np.mean(thermal_color_metrics[metric])
        ]
        stds = [
            np.std(optical_metrics[metric]),
            np.std(thermal_bw_metrics[metric]),
            np.std(thermal_color_metrics[metric])
        ]
        
        axes[i].bar(modalities, means, yerr=stds, capsize=10, alpha=0.7, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[i].set_title(f'{metric.capitalize()} Comparison', fontsize=14)
        axes[i].set_ylim(0, 1.0)
        axes[i].set_ylabel('Score', fontsize=12)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values on top of the bars
        for j, v in enumerate(means):
            axes[i].text(j, v + 0.03, f"{v:.3f}", ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['save_dir'], f"{CONFIG['which_model']}_metrics_comparison.png"))
    plt.show()
    
    
    def run_all_models():
        """
        Run all three model architectures and compare their performance.
        
        This function iterates through MobileNetV2, InceptionNetV3, and ResNet50,
        running the complete pipeline for each model and collecting results.
        It then creates a comprehensive comparison of model performance.
        
        Returns:
            dict: Results for all models and modalities
        """
        model_types = ["MobileNetV2", "InceptionNetV3", "ResNet50"]
        all_results = {}
        
        # Set appropriate image size for each model
        img_sizes = {
            "MobileNetV2": (224, 224),
            "InceptionNetV3": (299, 299),
            "ResNet50": (224, 224)
        }
        
        # Store original CONFIG settings
        original_config = CONFIG.copy()
        
        for model_type in model_types:
            print(f"\n\n{'='*50}")
            print(f"Starting evaluation with {model_type}")
            print(f"{'='*50}\n")
            
            # Update CONFIG with current model
            CONFIG['which_model'] = model_type
            CONFIG['img_size'] = img_sizes[model_type]
            
            # Run pipeline with this model
            results = run_pipeline()
            all_results[model_type] = results
        
        # Restore original CONFIG
        for key, value in original_config.items():
            CONFIG[key] = value
        
        # Compare results across different model architectures
        compare_model_architectures(all_results)
        
        return all_results

    def compare_model_architectures(all_results):
        """
        Compare performance across different model architectures.
        
        Creates a comprehensive comparison table and visualizations showing
        how each model architecture performs across different modalities and metrics.
        
        Args:
            all_results (dict): Dictionary containing results for all models and modalities
        """
        models = list(all_results.keys())
        modalities = ['optical', 'thermal_bw', 'thermal_color']
        metrics = ['accuracy', 'auc', 'sensitivity', 'specificity', 'f1']
        
        # Create a DataFrame to store the comparison
        comparison_data = []
        
        for model in models:
            for modality in modalities:
                results = all_results[model][modality]
                row = {
                    'Model': model,
                    'Modality': modality
                }
                
                for metric in metrics:
                    # Calculate mean and std
                    mean_val = np.mean(results[metric])
                    std_val = np.std(results[metric])
                    row[metric] = f"{mean_val:.3f}±{std_val:.3f}"
                
                comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\nComparison of model architectures across modalities:")
        print(comparison_df)
        
        # Save the comparison
        comparison_df.to_csv(os.path.join(CONFIG['save_dir'], "model_architecture_comparison.csv"), index=False)
        
        # Create visualization of the comparison
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            
            bar_width = 0.25
            index = np.arange(len(modalities))
            
            for i, model in enumerate(models):
                means = []
                stds = []
                
                for modality in modalities:
                    results = all_results[model][modality]
                    means.append(np.mean(results[metric]))
                    stds.append(np.std(results[metric]))
                
                plt.bar(index + i*bar_width, means, bar_width, yerr=stds, 
                    label=model, alpha=0.7)
            
            plt.xlabel('Modality')
            plt.ylabel(f'{metric.capitalize()} Score')
            plt.title(f'{metric.capitalize()} Comparison Across Models and Modalities')
            plt.xticks(index + bar_width, modalities)
            plt.ylim(0, 1.0)
            plt.legend(loc='best')
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Add values on top of the bars
            for i, model in enumerate(models):
                means = []
                for modality in modalities:
                    results = all_results[model][modality]
                    means.append(np.mean(results[metric]))
                
                for j, v in enumerate(means):
                    plt.text(j + i*bar_width, v + 0.02, f"{v:.3f}", 
                            ha='center', va='bottom', fontsize=8, rotation=0)
            
            plt.tight_layout()
            plt.savefig(os.path.join(CONFIG['save_dir'], f"model_comparison_{metric}.png"))
            plt.show()
        
        # Create a heatmap for best model per modality and metric
        best_models = pd.DataFrame(index=modalities, columns=metrics)
        
        for modality in modalities:
            for metric in metrics:
                best_value = 0
                best_model = ""
                
                for model in models:
                    value = np.mean(all_results[model][modality][metric])
                    if value > best_value:
                        best_value = value
                        best_model = model
                
                best_models.loc[modality, metric] = f"{best_model}\n({best_value:.3f})"
        
        # plt.figure(figsize=(12, 6))
        # sns.heatmap(best_models.notna(), annot=best_models, fmt='', cmap='YlGnBu', linewidths=0.5, cbar=False)
        # plt.title('Best Performing Model per Modality and Metric')
        # plt.tight_layout()
        # plt.savefig(os.path.join(CONFIG['save_dir'], "best_models_heatmap.png"))
        # plt.show()
        return {
        'optical': optical_metrics,
        'thermal_bw': thermal_bw_metrics,
        'thermal_color': thermal_color_metrics
    }



# Usage example in main block
if __name__ == "__main__":
    """
    Main execution block to run the erythema classification pipeline.
    
    To use this script:
    1. Modify the CONFIG dictionary above with your file paths
    2. Uncomment the run_pipeline() line to execute a single model (based on CONFIG)
    3. Uncomment the run_all_models() line to run and compare all three models
    """
    # You can customize these values before running
    CONFIG = {
        # File paths - MODIFY THESE WITH YOUR ACTUAL PATHS
        'excel_path': "/path/to/ErythemaImgLabels.xlsx",
        'erythema_path': "/path/to/Processed_Images/",
        'monk_filepath': "/path/to/Monk_skin_tone_groupings.xlsx",
        'save_dir': "./Saved_models",
        
        # Model configuration (default, will be overridden when running all models)
        'which_model': "MobileNetV2",
        'img_size': (224, 224),
        
        # Training parameters
        'batch_size': 32,
        'num_epochs': 100,
        'patience': 15,
        'seed': 42
    }
    
    # Uncomment ONE of the following lines:
    # results = run_pipeline()          # Run just one model (specified in CONFIG)
    # all_results = run_all_models()    # Run all three models and compare
