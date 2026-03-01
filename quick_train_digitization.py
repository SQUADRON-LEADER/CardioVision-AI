"""
Quick Training Script for ECG Digitization Model
Optimized for faster training with reduced epochs
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from ECG_Digitization_Model import AdvancedECGDigitizationModel, ECGDigitizationLoss
from ECG_Digitization_Dataset import SimpleECGDigitizationDataset

# Configuration
BATCH_SIZE = 8  # Smaller batch size for faster iteration
NUM_EPOCHS = 5  # Reduced for quick training
LEARNING_RATE = 0.0001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_DIR = 'checkpoints'
PTB_XL_PATH = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'

print(f"🚀 Starting ECG Digitization Training")
print(f"📱 Device: {DEVICE}")
print(f"🔢 Batch Size: {BATCH_SIZE}")
print(f"🔁 Epochs: {NUM_EPOCHS}")
print("="*60)

# Create checkpoint directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Initialize model
print("\n📦 Loading model...")
model = AdvancedECGDigitizationModel(signal_length=1000, num_leads=12, dropout=0.3)
model = model.to(DEVICE)
print(f"✅ Model loaded with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

# Create datasets
print("\n📊 Loading PTB-XL dataset...")
try:
    train_dataset = SimpleECGDigitizationDataset(
        ptbxl_dir=PTB_XL_PATH,
        num_samples=500,  # Use subset for quick training
        sampling_rate=100,
        signal_length=1000
    )
    
    val_dataset = SimpleECGDigitizationDataset(
        ptbxl_dir=PTB_XL_PATH,
        num_samples=100,  # Small validation set
        sampling_rate=100,
        signal_length=1000
    )
    
    print(f"✅ Train samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}")
    
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    print("\n💡 Tip: Ensure PTB-XL dataset is in the correct location")
    exit(1)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,  # Set to 0 for Windows compatibility
    pin_memory=True if DEVICE == 'cuda' else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True if DEVICE == 'cuda' else False
)

# Loss and optimizer
criterion = ECGDigitizationLoss(
    mse_weight=1.0,
    shape_weight=0.3,
    freq_weight=0.2
)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)

# Training history
history = {
    'train_loss': [],
    'val_loss': [],
    'val_mse': []
}

best_val_loss = float('inf')

# Training loop
print("\n🎯 Starting Training...")
print("="*60)

for epoch in range(NUM_EPOCHS):
    # Training phase
    model.train()
    train_loss = 0.0
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
    
    for batch_idx, (images, signals) in enumerate(train_pbar):
        images = images.to(DEVICE)
        signals = signals.to(DEVICE)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(images)
        loss, loss_dict = criterion(predictions, signals)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_train_loss = train_loss / len(train_loader)
    history['train_loss'].append(avg_train_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_mse = 0.0
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]  ")
        for images, signals in val_pbar:
            images = images.to(DEVICE)
            signals = signals.to(DEVICE)
            
            predictions = model(images)
            loss, loss_dict = criterion(predictions, signals)
            mse = nn.MSELoss()(predictions, signals)
            
            val_loss += loss.item()
            val_mse += mse.item()
            val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_mse = val_mse / len(val_loader)
    
    history['val_loss'].append(avg_val_loss)
    history['val_mse'].append(avg_val_mse)
    
    # Print epoch summary
    print(f"\n📊 Epoch {epoch+1} Summary:")
    print(f"   Train Loss: {avg_train_loss:.4f}")
    print(f"   Val Loss:   {avg_val_loss:.4f}")
    print(f"   Val MSE:    {avg_val_mse:.4f}")
    
    # Learning rate scheduling
    scheduler.step(avg_val_loss)
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model_digitization.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'val_mse': avg_val_mse
        }, checkpoint_path)
        print(f"   ✅ Best model saved! (Val Loss: {avg_val_loss:.4f})")
    
    print("="*60)

# Save final model
final_path = os.path.join(CHECKPOINT_DIR, f'digitization_model_epoch_{NUM_EPOCHS}.pth')
torch.save({
    'epoch': NUM_EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'history': history
}, final_path)

print("\n🎉 Training Complete!")
print(f"✅ Best model saved to: {os.path.join(CHECKPOINT_DIR, 'best_model_digitization.pth')}")
print(f"✅ Final model saved to: {final_path}")
print(f"📉 Best Val Loss: {best_val_loss:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss', marker='o')
plt.plot(history['val_loss'], label='Val Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['val_mse'], label='Val MSE', marker='o', color='red')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Validation MSE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plot_path = os.path.join(CHECKPOINT_DIR, 'training_history.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"📈 Training plot saved to: {plot_path}")

# Copy best model to flask_backend
import shutil
backend_model_path = 'flask_backend/best_model_digitization.pth'
try:
    shutil.copy(
        os.path.join(CHECKPOINT_DIR, 'best_model_digitization.pth'),
        backend_model_path
    )
    print(f"\n✅ Model copied to backend: {backend_model_path}")
    print("🚀 System is now ready for digitization!")
except Exception as e:
    print(f"\n⚠️  Could not copy to backend: {e}")
    print(f"   Please manually copy:")
    print(f"   copy {os.path.join(CHECKPOINT_DIR, 'best_model_digitization.pth')} {backend_model_path}")

print("\n" + "="*60)
print("🎯 Next Steps:")
print("1. Start Flask backend: cd flask_backend && python app.py")
print("2. Open frontend: frontend/index_enhanced.html")
print("3. Test digitization mode with ECG images")
print("="*60)
