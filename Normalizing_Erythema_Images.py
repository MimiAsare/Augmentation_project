"""
Author: Miriam Asare-Baiden
Date: 12/06/2024
Purpose: Normalize Phase 2 images using global temperature extremes (min/max across all images) via Minmax normalization, and 
         save  the normalized images to a new directory.

"""



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_min_max_temperature(original_temperature_folder):
    """
    Compute the minimum and maximum temperature values from CSV images in the specified folder.
    
    Parameters:
    original_temperature_folder (str): Path to the folder containing temperature CSV files.
    
    Returns:
    tuple: (updated_min, updated_max) representing the minimum and maximum temperature values found.
    """
    updated_min = float('inf')
    updated_max = float('-inf')

    # Load erythema files once
    erythema_files = pd.read_excel(
        r'./SCRG/HIP_Project/ProcessedData/ErythemaImgLabels.xlsx'
    )
    erythema_names = erythema_files['Img Name'].str.split('.').str[0]
    
    for root, _, files in os.walk(original_temperature_folder):
        for file in files:
            file_path = os.path.join(root, file)
            if not os.path.isfile(file_path) or not file.endswith('.csv'):
                continue
            
            actual_filename = os.path.splitext(file)[0]
            if not actual_filename.startswith('HIP_'):
                print(f"Skipping file {file} as it doesn't match HIP format")
                continue

            if actual_filename not in erythema_names.values:
                continue

            try:
                df = pd.read_csv(file_path, header=None)
                temperature_values = df.values.flatten()
                current_min = np.min(temperature_values)
                current_max = np.max(temperature_values)
                
                updated_min = min(updated_min, current_min)
                updated_max = max(updated_max, current_max)
            except Exception as e:
                print(f"Error processing file {file}: {e}")    

    return updated_min, updated_max

def compare_images(temperature_folder, normalized_temperature_folder):
    """
    Normalize temperature values from CSV files and save the results.
    
    Parameters:
    temperature_folder (str): Path to the folder containing original temperature CSV files.
    normalized_temperature_folder (str): Path to save normalized temperature CSV files.
    """
    min_temp, max_temp = get_min_max_temperature(temperature_folder)
    print(f'Minimum temperature across files: {min_temp}')
    print(f'Maximum temperature across files: {max_temp}\n')

    for root, _, files in os.walk(temperature_folder):
        for file in files:
            if not file.endswith('.csv'):
                continue
            
            file_path = os.path.join(root, file)
            if not os.path.isfile(file_path):
                continue
            
            actual_filename = os.path.splitext(file)[0]
            file_parts = actual_filename.split('_')
            hip_folder_name = "_".join(file_parts[:2])
            
            if not actual_filename.startswith('HIP_'):
                print(f"Skipping file {file} as it doesn't match HIP format")
                continue
            
            # Assuming ErythemaImgLabels.xlsx is in the same folder(Update with the  actual erythema file path)
            erythema_files = pd.read_excel(
                r'./SCRG/HIP_Project/ProcessedData/ErythemaImgLabels.xlsx'
            )
            erythema_names = erythema_files['Img Name'].str.split('.').str[0]
            
            if actual_filename not in erythema_names.values:
                continue
            
            try:
                original_array = pd.read_csv(file_path, header=None)
                original_array_values = pd.to_numeric(original_array.values.flatten(), errors='coerce')
                
                normalized = ((original_array_values - min_temp) * 255) / (max_temp - min_temp)
                normalized_array_values = normalized.reshape(240, 320)
                
                hip_folder = os.path.join(normalized_temperature_folder, hip_folder_name)
                os.makedirs(hip_folder, exist_ok=True)
                
                output_path = os.path.join(hip_folder, f'{actual_filename}_normalized.csv')
                normalized_df = pd.DataFrame(normalized_array_values)
                normalized_df.to_csv(output_path, index=False, header=None)
                
                print(f"Processed and saved: {output_path}")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                ax1.imshow(original_array, cmap='gray')
                ax1.set_title(f'Original Image_{actual_filename}')
                ax1.axis('off')
                
                ax2.imshow(normalized_array_values, cmap='gray')
                ax2.set_title(f'Normalized Image_{actual_filename}')
                ax2.axis('off')
                
                save_img_path = os.path.join(
                    '/Users/macbookpro/Library/CloudStorage/OneDrive-Emory/Research_projects/Pressure_injury/Thermography/Phase_2/Normalized_images/',
                    f'{actual_filename}.png'
                )
                plt.savefig(save_img_path)
                print(f'Normalized {actual_filename} saved successfully to {save_img_path}')
                plt.tight_layout()
                plt.close()
            except Exception as e:
                print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    original_temp_path = r'./SCRG/HIP_Project/ProcessedData/Separated & Processed Images/temperature_csv/' # Update path
    normalized_temp_path = ''# add path to save normalized images
    compare_images(original_temp_path, normalized_temp_path)
