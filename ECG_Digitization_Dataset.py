"""
ECG Digitization Dataset
Loads paired ECG images and corresponding time-series signals from PTB-XL dataset
"""

import os
import numpy as np
import pandas as pd
import wfdb
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
from pathlib import Path
import ast


class ECGDigitizationDataset(Dataset):
    """
    Dataset for ECG image-to-signal digitization
    Loads both ECG images and their corresponding time-series signals
    """
    
    def __init__(
        self,
        ptbxl_dir,
        image_dir,
        sampling_rate=100,  # PTB-XL has 100Hz and 500Hz, use 100Hz for efficiency
        signal_length=1000,  # 10 seconds at 100Hz
        transform=None,
        augment=False
    ):
        """
        Args:
            ptbxl_dir: Path to PTB-XL dataset directory
            image_dir: Path to generated ECG images directory
            sampling_rate: 100 or 500 Hz
            signal_length: Number of samples per signal
            transform: Image transforms
            augment: Apply data augmentation
        """
        self.ptbxl_dir = ptbxl_dir
        self.image_dir = image_dir
        self.sampling_rate = sampling_rate
        self.signal_length = signal_length
        self.transform = transform
        self.augment = augment
        
        # Load PTB-XL database
        self.database = pd.read_csv(os.path.join(ptbxl_dir, 'ptbxl_database.csv'))
        
        # Filter by sampling rate
        self.database = self.database[self.database['sampling_rate'] == sampling_rate]
        
        # Build paired data list
        self.data_pairs = []
        self._build_data_pairs()
        
        print(f"Loaded {len(self.data_pairs)} paired samples")
        
    def _build_data_pairs(self):
        """Build list of valid image-signal pairs"""
        for idx, row in self.database.iterrows():
            # Get signal path
            if self.sampling_rate == 100:
                signal_path = os.path.join(self.ptbxl_dir, row['filename_lr'])
            else:
                signal_path = os.path.join(self.ptbxl_dir, row['filename_hr'])
            
            # Get image path (generated from signal)
            record_id = Path(row['filename_lr']).stem
            
            # Check multiple possible image locations
            possible_image_paths = [
                os.path.join(self.image_dir, 'normal_ecg_images', f'{record_id}.png'),
                os.path.join(self.image_dir, 'abnormal_heartbeat_ecg_images', f'{record_id}.png'),
                os.path.join(self.image_dir, 'myocardial_infarction_ecg_images', f'{record_id}.png'),
                os.path.join(self.image_dir, 'post_mi_history_ecg_images', f'{record_id}.png'),
            ]
            
            # Find existing image
            image_path = None
            for path in possible_image_paths:
                if os.path.exists(path):
                    image_path = path
                    break
            
            # Only add if both image and signal exist
            if image_path and os.path.exists(signal_path + '.hea'):
                self.data_pairs.append({
                    'image_path': image_path,
                    'signal_path': signal_path,
                    'record_id': record_id,
                    'metadata': row
                })
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: (3, 224, 224) tensor
            signal: (12, signal_length) tensor - normalized ECG signals
            metadata: dict with additional info
        """
        pair = self.data_pairs[idx]
        
        # Load image
        try:
            image = Image.open(pair['image_path']).convert('RGB')
            image = image.resize((224, 224), Image.LANCZOS)
            
            # Preprocess image (contrast enhancement)
            image = self._preprocess_image(image)
            
            # Augmentation
            if self.augment:
                image = self._augment_image(image)
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            else:
                # Default: convert to tensor and normalize
                image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        except Exception as e:
            print(f"Error loading image {pair['image_path']}: {e}")
            # Return dummy image
            image = torch.zeros(3, 224, 224)
        
        # Load signal
        try:
            record = wfdb.rdrecord(pair['signal_path'])
            signal = record.p_signal  # Shape: (samples, 12)
            
            # Transpose to (12, samples)
            signal = signal.T
            
            # Normalize signal length
            signal = self._normalize_signal_length(signal)
            
            # Normalize amplitude
            signal = self._normalize_signal_amplitude(signal)
            
            # Convert to tensor
            signal = torch.from_numpy(signal).float()
        
        except Exception as e:
            print(f"Error loading signal {pair['signal_path']}: {e}")
            # Return dummy signal
            signal = torch.zeros(12, self.signal_length)
        
        # Metadata
        metadata = {
            'record_id': pair['record_id'],
            'image_path': pair['image_path'],
            'signal_path': pair['signal_path']
        }
        
        return image, signal, metadata
    
    def _preprocess_image(self, image):
        """Apply CLAHE contrast enhancement"""
        img_array = np.array(image)
        
        # Convert to LAB color space
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge back
        enhanced = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(enhanced)
    
    def _augment_image(self, image):
        """Apply data augmentation"""
        # Random horizontal flip
        if np.random.rand() < 0.2:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Random rotation
        if np.random.rand() < 0.2:
            angle = np.random.uniform(-3, 3)
            image = image.rotate(angle, fillcolor=(255, 255, 255))
        
        # Random brightness/contrast
        if np.random.rand() < 0.3:
            img_array = np.array(image).astype(np.float32)
            brightness = np.random.uniform(0.9, 1.1)
            contrast = np.random.uniform(0.9, 1.1)
            img_array = img_array * contrast * brightness
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            image = Image.fromarray(img_array)
        
        return image
    
    def _normalize_signal_length(self, signal):
        """
        Normalize signal to fixed length
        
        Args:
            signal: (12, original_length)
        Returns:
            normalized: (12, signal_length)
        """
        current_length = signal.shape[1]
        
        if current_length == self.signal_length:
            return signal
        elif current_length > self.signal_length:
            # Truncate
            return signal[:, :self.signal_length]
        else:
            # Pad with zeros
            pad_width = ((0, 0), (0, self.signal_length - current_length))
            return np.pad(signal, pad_width, mode='constant', constant_values=0)
    
    def _normalize_signal_amplitude(self, signal):
        """
        Normalize signal amplitude to [-1, 1] range
        
        Args:
            signal: (12, signal_length)
        Returns:
            normalized: (12, signal_length)
        """
        # Normalize per lead
        normalized = np.zeros_like(signal)
        
        for lead_idx in range(signal.shape[0]):
            lead_signal = signal[lead_idx]
            
            # Remove baseline
            lead_signal = lead_signal - np.mean(lead_signal)
            
            # Normalize to [-1, 1] using robust percentile scaling
            p_low = np.percentile(lead_signal, 1)
            p_high = np.percentile(lead_signal, 99)
            
            if p_high - p_low > 1e-6:
                lead_signal = 2 * (lead_signal - p_low) / (p_high - p_low) - 1
                lead_signal = np.clip(lead_signal, -1, 1)
            
            normalized[lead_idx] = lead_signal
        
        return normalized


class SimpleECGDigitizationDataset(Dataset):
    """
    Simplified version that generates synthetic paired data
    Useful for testing when you don't have pre-generated images
    """
    
    def __init__(
        self,
        ptbxl_dir,
        num_samples=1000,
        sampling_rate=100,
        signal_length=1000,
        transform=None
    ):
        """
        Args:
            ptbxl_dir: Path to PTB-XL dataset
            num_samples: Number of samples to load
            sampling_rate: 100 or 500 Hz
            signal_length: Number of samples per signal
            transform: Image transforms
        """
        self.ptbxl_dir = ptbxl_dir
        self.sampling_rate = sampling_rate
        self.signal_length = signal_length
        self.transform = transform
        
        # Load database
        self.database = pd.read_csv(os.path.join(ptbxl_dir, 'ptbxl_database.csv'))
        
        # PTB-XL has both 100Hz and 500Hz files, but no sampling_rate column
        # We'll use all records and select the appropriate filename based on sampling_rate
        
        # Limit samples
        self.database = self.database.head(num_samples)
        
        print(f"Loaded {len(self.database)} samples at {sampling_rate}Hz")
    
    def __len__(self):
        return len(self.database)
    
    def __getitem__(self, idx):
        """Returns dynamically generated image and signal"""
        row = self.database.iloc[idx]
        
        # Load signal
        if self.sampling_rate == 100:
            signal_path = os.path.join(self.ptbxl_dir, row['filename_lr'])
        else:
            signal_path = os.path.join(self.ptbxl_dir, row['filename_hr'])
        
        try:
            record = wfdb.rdrecord(signal_path)
            signal = record.p_signal.T  # (12, samples)
            
            # Normalize signal
            signal = self._normalize_signal(signal)
            
            # Generate image from signal
            pil_image = self._signal_to_image(signal)
            
            # Convert PIL Image to tensor
            if self.transform:
                image = self.transform(pil_image)
            else:
                # Default: convert to tensor and normalize
                image = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float() / 255.0
            
            signal_tensor = torch.from_numpy(signal[:, :self.signal_length]).float()
            
            return image, signal_tensor
        
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            return torch.zeros(3, 224, 224), torch.zeros(12, self.signal_length)
    
    def _normalize_signal(self, signal):
        """Normalize signal amplitude"""
        normalized = np.zeros_like(signal)
        for i in range(12):
            lead = signal[i]
            lead = lead - np.mean(lead)
            p_low, p_high = np.percentile(lead, [1, 99])
            if p_high - p_low > 1e-6:
                lead = 2 * (lead - p_low) / (p_high - p_low) - 1
                lead = np.clip(lead, -1, 1)
            normalized[i] = lead
        return normalized
    
    def _signal_to_image(self, signal, width=1000, height=800):
        """
        Convert signal to ECG-like image
        
        Args:
            signal: (12, samples) numpy array
        Returns:
            image: PIL Image (224, 224)
        """
        # Create canvas
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Draw grid
        grid_color = (220, 220, 220)
        for i in range(0, width, 20):
            cv2.line(img, (i, 0), (i, height), grid_color, 1)
        for i in range(0, height, 20):
            cv2.line(img, (0, i), (width, i), grid_color, 1)
        
        # Draw each lead
        n_leads = min(signal.shape[0], 12)
        lead_height = height // n_leads
        
        for lead_idx in range(n_leads):
            lead_signal = signal[lead_idx]
            
            # Limit to display width
            n_samples = min(len(lead_signal), width)
            lead_signal = lead_signal[:n_samples]
            
            # Scale to lead height
            y_offset = lead_idx * lead_height + lead_height // 2
            y_scale = lead_height * 0.4  # Use 40% of available height
            
            # Draw signal
            for i in range(n_samples - 1):
                x1, y1 = i, int(y_offset - lead_signal[i] * y_scale)
                x2, y2 = i + 1, int(y_offset - lead_signal[i + 1] * y_scale)
                
                # Clip to valid range
                y1 = np.clip(y1, 0, height - 1)
                y2 = np.clip(y2, 0, height - 1)
                
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 1)
        
        # Convert to PIL and resize
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((224, 224), Image.LANCZOS)
        
        return pil_img


# ==== UTILITY FUNCTIONS ====

def create_digitization_dataloaders(
    ptbxl_dir,
    image_dir=None,
    batch_size=16,
    num_workers=4,
    train_split=0.8,
    sampling_rate=100,
    signal_length=1000
):
    """
    Create train/val dataloaders for ECG digitization
    
    Args:
        ptbxl_dir: Path to PTB-XL dataset
        image_dir: Path to generated images (if None, use SimpleDataset)
        batch_size: Batch size
        num_workers: Number of data loading workers
        train_split: Train/val split ratio
        sampling_rate: Signal sampling rate
        signal_length: Signal length
    
    Returns:
        train_loader, val_loader
    """
    from torchvision import transforms
    from torch.utils.data import DataLoader, random_split
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    if image_dir and os.path.exists(image_dir):
        print(f"Using paired image-signal dataset from {image_dir}")
        dataset = ECGDigitizationDataset(
            ptbxl_dir=ptbxl_dir,
            image_dir=image_dir,
            sampling_rate=sampling_rate,
            signal_length=signal_length,
            transform=transform,
            augment=True
        )
    else:
        print("Using dynamically generated images (SimpleDataset)")
        dataset = SimpleECGDigitizationDataset(
            ptbxl_dir=ptbxl_dir,
            num_samples=2000,  # Limit for testing
            sampling_rate=sampling_rate,
            signal_length=signal_length,
            transform=transform
        )
    
    # Split into train/val
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    print("Testing ECG Digitization Dataset...")
    print("=" * 70)
    
    # Update this path to your PTB-XL location
    ptbxl_dir = r"ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    
    # Test SimpleDataset
    dataset = SimpleECGDigitizationDataset(
        ptbxl_dir=ptbxl_dir,
        num_samples=10,
        sampling_rate=100,
        signal_length=1000
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Load one sample
    image, signal, metadata = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Signal shape: {signal.shape}")
    print(f"Record ID: {metadata['record_id']}")
    
    print("=" * 70)
    print("✓ Dataset test passed!")
