"""
Convert PTB-XL ECG signals to images for digitization training
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import wfdb
from tqdm import tqdm
import ast

# Set paths
BASE_PATH = r'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
OUTPUT_BASE = r'ecg data new version'

# Create output directories
output_dirs = {
    'normal': os.path.join(OUTPUT_BASE, 'normal_ecg_images'),
    'abnormal_heartbeat': os.path.join(OUTPUT_BASE, 'abnormal_heartbeat_ecg_images'),
    'myocardial_infarction': os.path.join(OUTPUT_BASE, 'myocardial_infarction_ecg_images'),
    'post_mi_history': os.path.join(OUTPUT_BASE, 'post_mi_history_ecg_images')
}

for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

print("Loading PTB-XL database...")
# Load database
df = pd.read_csv(os.path.join(BASE_PATH, 'ptbxl_database.csv'))
print(f"Total records: {len(df)}")

# Parse scp_codes
df['scp_codes_dict'] = df.scp_codes.apply(lambda x: ast.literal_eval(x))

def categorize_ecg(scp_codes_dict):
    """Categorize ECG based on diagnostic codes"""
    codes = list(scp_codes_dict.keys())
    
    # Check for myocardial infarction
    mi_codes = ['IMI', 'AMI', 'ASMI', 'ILMI', 'ALMI', 'INJAS', 'LMI', 
                'INJAL', 'IPLMI', 'IPMI', 'INJIN', 'INJLA', 'PMI', 'INJIL']
    if any(code in codes for code in mi_codes):
        return 'myocardial_infarction'
    
    # Check for normal
    if 'NORM' in codes:
        return 'normal'
    
    # Check for rhythm abnormalities
    rhythm_codes = ['AFIB', 'AFLT', 'STACH', 'SBRAD', 'SARRH', 'SVTAC', 'BIGU', 'TRIGU']
    if any(code in codes for code in rhythm_codes):
        return 'abnormal_heartbeat'
    
    # Check for conduction blocks and other abnormalities
    cd_codes = ['1AVB', '2AVB', '3AVB', 'CRBBB', 'CLBBB', 'IRBBB', 'ILBBB', 'IVCD', 'WPW']
    if any(code in codes for code in cd_codes):
        return 'abnormal_heartbeat'
    
    # Default to abnormal heartbeat for other conditions
    return 'abnormal_heartbeat'

def plot_ecg_to_image(signal, filename, lead_names=None):
    """Convert ECG signal array to image file"""
    if lead_names is None:
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    n_leads = signal.shape[1]
    fig, axes = plt.subplots(n_leads, 1, figsize=(10, 8), sharex=True, dpi=80)
    fig.patch.set_facecolor('white')
    
    if n_leads == 1:
        axes = [axes]
    
    for i in range(n_leads):
        axes[i].plot(signal[:, i], linewidth=0.5, color='black')
        axes[i].set_ylabel(lead_names[i] if i < len(lead_names) else f'Lead {i+1}', 
                          fontsize=7)
        axes[i].grid(True, linewidth=0.3, alpha=0.5, color='red', linestyle='-')
        axes[i].set_ylim([signal[:, i].min() - 0.3, signal[:, i].max() + 0.3])
        axes[i].tick_params(labelsize=6)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
    
    axes[-1].set_xlabel('Samples', fontsize=7)
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(filename, dpi=80, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close('all')
    plt.clf()

print("\nCategorizing ECGs...")
df['category'] = df['scp_codes_dict'].apply(categorize_ecg)

# Count categories
print("\nCategory distribution:")
print(df['category'].value_counts())

# Limit samples per category for balanced dataset
max_samples_per_category = 500
print(f"\nLimiting to {max_samples_per_category} samples per category for balanced training...")

category_counts = {cat: 0 for cat in output_dirs.keys()}
total_converted = 0

print("\nConverting ECG signals to images...")
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing ECGs"):
    category = row['category']
    
    # Skip if we have enough samples for this category
    if category_counts[category] >= max_samples_per_category:
        continue
    
    try:
        # Load ECG signal (using 100Hz version for faster processing)
        signal_path = os.path.join(BASE_PATH, row['filename_lr'])
        signal, fields = wfdb.rdsamp(signal_path)
        
        # Generate output filename
        ecg_id = row['ecg_id']
        output_filename = os.path.join(output_dirs[category], f'ecg_{ecg_id:05d}.png')
        
        # Convert to image
        plot_ecg_to_image(signal, output_filename, fields['sig_name'])
        
        category_counts[category] += 1
        total_converted += 1
        
    except Exception as e:
        print(f"\nError processing ECG {row['ecg_id']}: {e}")
        continue
    
    # Break if we have enough samples in all categories
    if all(count >= max_samples_per_category for count in category_counts.values()):
        break

print("\n" + "="*70)
print("CONVERSION COMPLETE!")
print("="*70)
print("\nImages saved:")
for category, count in category_counts.items():
    print(f"{category:25s}: {count:4d} images")
print(f"\nTotal images created: {total_converted}")
print("="*70)
print("\n✓ Ready to train the model!")
