"""
Convert PTB-XL ECG signals to images - Simple & Fast Version
Uses OpenCV for robust image generation
"""
import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import ast
import wfdb

# Set paths
BASE_PATH = r'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
OUTPUT_BASE = r'ecg_data'

output_dirs = {
    'normal': os.path.join(OUTPUT_BASE, 'normal_ecg_images'),
    'abnormal_heartbeat': os.path.join(OUTPUT_BASE, 'abnormal_heartbeat_ecg_images'),
    'myocardial_infarction': os.path.join(OUTPUT_BASE, 'myocardial_infarction_ecg_images'),
    'post_mi_history': os.path.join(OUTPUT_BASE, 'post_mi_history_ecg_images')
}

# Create output directories explicitly
for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)
    print(f"Created directory: {os.path.abspath(dir_path)}")

def create_ecg_image_opencv(signal, output_file, width=1000, height=800):
    """Create ECG image using OpenCV - fast and robust"""
    n_leads = signal.shape[1]
    lead_height = height // n_leads
    
    # Create white background
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw grid (like ECG paper)
    grid_color = (255, 200, 200)  # Light red
    for i in range(0, width, 20):
        cv2.line(img, (i, 0), (i, height), grid_color, 1)
    for i in range(0, height, 20):
        cv2.line(img, (0, i), (width, i), grid_color, 1)
    
    # Draw each lead
    for lead_idx in range(n_leads):
        lead_signal = signal[:, lead_idx]
        
        # Normalize signal to fit in lead space        
        signal_min, signal_max = lead_signal.min(), lead_signal.max()
        signal_range = signal_max - signal_min
        if signal_range == 0:
            signal_range = 1
        
        y_offset = lead_idx * lead_height + lead_height // 2
        y_scale = (lead_height * 0.7) / signal_range
        
        # Draw signal
        points = []
        for i, val in enumerate(lead_signal):
            x = int((i / len(lead_signal)) * width)
            y = int(y_offset - (val - signal_min - signal_range/2) * y_scale)
            y = max(lead_idx * lead_height + 10, min(y, (lead_idx + 1) * lead_height - 10))
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        cv2.polylines(img, [points], False, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Add lead label
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        label = lead_names[lead_idx] if lead_idx < len(lead_names) else f'L{lead_idx+1}'
        cv2.putText(img, label, (10, y_offset - lead_height//3), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Save image
    cv2.imwrite(output_file, img)

def categorize_ecg(scp_codes_dict):
    """Categorize ECG based on diagnostic codes"""
    codes = list(scp_codes_dict.keys())
    
    mi_codes = ['IMI', 'AMI', 'ASMI', 'ILMI', 'ALMI', 'INJAS', 'LMI', 
                'INJAL', 'IPLMI', 'IPMI', 'INJIN', 'INJLA', 'PMI', 'INJIL']
    if any(code in codes for code in mi_codes):
        return 'myocardial_infarction'
    
    if 'NORM' in codes:
        return 'normal'
    
    return 'abnormal_heartbeat'

print("Loading PTB-XL database...")
df = pd.read_csv(os.path.join(BASE_PATH, 'ptbxl_database.csv'))
df['scp_codes_dict'] = df.scp_codes.apply(lambda x: ast.literal_eval(x))
df['category'] = df['scp_codes_dict'].apply(categorize_ecg)

print(f"\nTotal records: {len(df)}")
print("\nCategory distribution:")
print(df['category'].value_counts())

# Target: 2000 images per category for better accuracy
max_samples = 2000
category_counts = {cat: 0 for cat in output_dirs.keys()}

# Count existing images
for cat, path in output_dirs.items():
    if os.path.exists(path):
        existing = len([f for f in os.listdir(path) if f.endswith('.png')])
        category_counts[cat] = existing
        print(f"{cat}: {existing} images already exist")

print(f"\nGenerating images (target: {max_samples} per category)...")

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
    category = row['category']
    
    # Skip if we have enough for this category
    if category_counts[category] >= max_samples:
        continue
    
    # Check if all categories are complete
    if all(count >= max_samples for count in category_counts.values()):
        break
    
    try:
        # Load ECG signal
        signal_path = os.path.join(BASE_PATH, row['filename_lr'])
        signal, _ = wfdb.rdsamp(signal_path)
        
        # Create output filename
        ecg_id = row['ecg_id']
        output_file = os.path.join(output_dirs[category], f'ecg_{ecg_id:05d}.png')
        
        # Skip if already exists
        if os.path.exists(output_file):
            category_counts[category] += 1
            continue
        
        # Create image
        create_ecg_image_opencv(signal, output_file)
        category_counts[category] += 1
        
    except Exception as e:
        continue

print("\n" + "="*70)
print("CONVERSION COMPLETE!")
print("="*70)
for cat, count in category_counts.items():
    print(f"{cat:25s}: {count:4d} images")
print(f"Total: {sum(category_counts.values())} images")
print("="*70)
