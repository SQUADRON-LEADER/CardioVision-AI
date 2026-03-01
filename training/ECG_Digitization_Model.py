"""
ECG Image Digitization Model
Converts ECG images to 12-lead time-series signals

Architecture: CNN Encoder + Transformer Decoder
- Input: ECG image (224x224x3)
- Output: 12 leads × 1000 samples (500Hz, 2 seconds)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ECGDigitizationModel(nn.Module):
    """
    End-to-end ECG digitization model
    Converts ECG images to time-series signals
    """
    
    def __init__(self, signal_length=1000, num_leads=12, dropout=0.3):
        super(ECGDigitizationModel, self).__init__()
        
        self.signal_length = signal_length
        self.num_leads = num_leads
        
        # ==== ENCODER: Extract features from ECG image ====
        # Use pretrained ResNet50 as backbone
        resnet = models.resnet50(pretrained=True)
        
        # Remove final FC layer
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # Output: (B, 2048, 7, 7)
        
        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # (B, 2048, 4, 4)
        
        # Flatten features
        self.flatten = nn.Flatten()  # (B, 2048*4*4) = (B, 32768)
        
        # Feature compression
        self.feature_compress = nn.Sequential(
            nn.Linear(2048 * 4 * 4, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ==== DECODER: Generate time-series signals ====
        
        # Signal projection layers (one per lead)
        self.signal_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, signal_length),
                nn.Tanh()  # Output in range [-1, 1]
            ) for _ in range(num_leads)
        ])
        
        # Refinement layer using 1D convolutions
        self.refinement = nn.Sequential(
            nn.Conv1d(num_leads, num_leads, kernel_size=15, padding=7, groups=num_leads),
            nn.BatchNorm1d(num_leads),
            nn.ReLU(),
            nn.Conv1d(num_leads, num_leads, kernel_size=15, padding=7, groups=num_leads),
            nn.BatchNorm1d(num_leads),
            nn.Tanh()
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input image tensor (B, 3, 224, 224)
            
        Returns:
            signals: Digitized ECG signals (B, 12, 1000)
        """
        # Encode image to features
        features = self.encoder(x)  # (B, 2048, 7, 7)
        features = self.adaptive_pool(features)  # (B, 2048, 4, 4)
        features = self.flatten(features)  # (B, 32768)
        features = self.feature_compress(features)  # (B, 1024)
        
        # Decode features to signals (one lead at a time)
        lead_signals = []
        for i in range(self.num_leads):
            lead_signal = self.signal_projections[i](features)  # (B, 1000)
            lead_signals.append(lead_signal)
        
        # Stack leads
        signals = torch.stack(lead_signals, dim=1)  # (B, 12, 1000)
        
        # Refine signals with 1D convolutions
        signals = self.refinement(signals)  # (B, 12, 1000)
        
        return signals


class AdvancedECGDigitizationModel(nn.Module):
    """
    Advanced ECG digitization with attention and residual connections
    Better quality signal reconstruction
    """
    
    def __init__(self, signal_length=1000, num_leads=12, dropout=0.3):
        super(AdvancedECGDigitizationModel, self).__init__()
        
        self.signal_length = signal_length
        self.num_leads = num_leads
        
        # ==== ENCODER ====
        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        # Multi-scale feature extraction
        self.multiscale_pool = nn.ModuleList([
            nn.AdaptiveAvgPool2d((2, 2)),  # (B, 2048, 2, 2)
            nn.AdaptiveAvgPool2d((4, 4)),  # (B, 2048, 4, 4)
            nn.AdaptiveAvgPool2d((8, 8))   # (B, 2048, 8, 8)
        ])
        
        # Feature fusion
        multiscale_features_size = 2048 * (2*2 + 4*4 + 8*8)  # 2048 * 84 = 172032
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(multiscale_features_size, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ==== ATTENTION MECHANISM ====
        self.lead_attention = nn.MultiheadAttention(
            embed_dim=2048,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # ==== DECODER ====
        
        # Lead-specific decoders with residual connections
        self.lead_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, signal_length)
            ) for _ in range(num_leads)
        ])
        
        # Signal refinement network
        self.refinement_network = nn.Sequential(
            # First refinement layer
            nn.Conv1d(num_leads, num_leads * 2, kernel_size=31, padding=15),
            nn.BatchNorm1d(num_leads * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Second refinement layer
            nn.Conv1d(num_leads * 2, num_leads * 2, kernel_size=15, padding=7),
            nn.BatchNorm1d(num_leads * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Output layer
            nn.Conv1d(num_leads * 2, num_leads, kernel_size=7, padding=3),
            nn.Tanh()
        )
        
    def forward(self, x):
        """
        Forward pass with attention
        
        Args:
            x: Input image tensor (B, 3, 224, 224)
            
        Returns:
            signals: Digitized ECG signals (B, 12, 1000)
        """
        batch_size = x.size(0)
        
        # Encode image
        features = self.encoder(x)  # (B, 2048, H, W)
        
        # Multi-scale pooling
        multiscale_features = []
        for pool in self.multiscale_pool:
            pooled = pool(features)  # (B, 2048, h, w)
            multiscale_features.append(pooled.flatten(1))  # (B, 2048*h*w)
        
        # Concatenate multi-scale features
        combined_features = torch.cat(multiscale_features, dim=1)  # (B, 172032)
        
        # Fuse features
        fused_features = self.feature_fusion(combined_features)  # (B, 2048)
        
        # Prepare for attention (expand to 12 leads)
        lead_features = fused_features.unsqueeze(1).repeat(1, self.num_leads, 1)  # (B, 12, 2048)
        
        # Apply attention
        attended_features, _ = self.lead_attention(
            lead_features, lead_features, lead_features
        )  # (B, 12, 2048)
        
        # Decode each lead
        lead_signals = []
        for i in range(self.num_leads):
            lead_feature = attended_features[:, i, :]  # (B, 2048)
            lead_signal = self.lead_decoders[i](lead_feature)  # (B, 1000)
            lead_signals.append(lead_signal)
        
        # Stack leads
        signals = torch.stack(lead_signals, dim=1)  # (B, 12, 1000)
        
        # Refine signals
        refined_signals = self.refinement_network(signals)  # (B, 12, 1000)
        
        return refined_signals


class SignalQualityPredictor(nn.Module):
    """
    Auxiliary network to predict signal quality
    Helps the main model learn better representations
    """
    
    def __init__(self, signal_length=1000, num_leads=12):
        super(SignalQualityPredictor, self).__init__()
        
        self.quality_network = nn.Sequential(
            nn.Conv1d(num_leads, 32, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Quality score 0-1
        )
    
    def forward(self, signals):
        """
        Args:
            signals: (B, 12, 1000)
        Returns:
            quality_score: (B, 1) - quality score between 0 and 1
        """
        return self.quality_network(signals)


# ==== LOSS FUNCTIONS ====

class ECGDigitizationLoss(nn.Module):
    """
    Combined loss for ECG digitization:
    - MSE: Point-wise reconstruction
    - Shape loss: Derivative matching
    - Frequency loss: Spectral similarity
    """
    
    def __init__(self, mse_weight=1.0, shape_weight=0.3, freq_weight=0.2):
        super(ECGDigitizationLoss, self).__init__()
        self.mse_weight = mse_weight
        self.shape_weight = shape_weight
        self.freq_weight = freq_weight
        
    def forward(self, pred_signals, true_signals):
        """
        Args:
            pred_signals: (B, 12, 1000) - predicted signals
            true_signals: (B, 12, 1000) - ground truth signals
        """
        # 1. MSE Loss (point-wise)
        mse_loss = F.mse_loss(pred_signals, true_signals)
        
        # 2. Shape Loss (derivative matching)
        pred_diff = pred_signals[:, :, 1:] - pred_signals[:, :, :-1]
        true_diff = true_signals[:, :, 1:] - true_signals[:, :, :-1]
        shape_loss = F.mse_loss(pred_diff, true_diff)
        
        # 3. Frequency Loss (FFT similarity)
        pred_fft = torch.fft.rfft(pred_signals, dim=2)
        true_fft = torch.fft.rfft(true_signals, dim=2)
        freq_loss = F.mse_loss(torch.abs(pred_fft), torch.abs(true_fft))
        
        # Combined loss
        total_loss = (
            self.mse_weight * mse_loss +
            self.shape_weight * shape_loss +
            self.freq_weight * freq_loss
        )
        
        return total_loss, {
            'mse': mse_loss.item(),
            'shape': shape_loss.item(),
            'freq': freq_loss.item(),
            'total': total_loss.item()
        }


# ==== MODEL FACTORY ====

def create_digitization_model(model_type='basic', **kwargs):
    """
    Factory function to create digitization models
    
    Args:
        model_type: 'basic' or 'advanced'
        **kwargs: Model parameters
    
    Returns:
        model: ECG digitization model
    """
    if model_type == 'basic':
        model = ECGDigitizationModel(**kwargs)
    elif model_type == 'advanced':
        model = AdvancedECGDigitizationModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


if __name__ == "__main__":
    # Test model
    print("Testing ECG Digitization Models...")
    print("=" * 70)
    
    # Test basic model
    model_basic = ECGDigitizationModel(signal_length=1000, num_leads=12)
    x = torch.randn(2, 3, 224, 224)
    y = model_basic(x)
    print(f"Basic Model - Input shape: {x.shape}, Output shape: {y.shape}")
    
    # Count parameters
    params = sum(p.numel() for p in model_basic.parameters() if p.requires_grad)
    print(f"Basic Model - Parameters: {params:,}")
    
    print()
    
    # Test advanced model
    model_advanced = AdvancedECGDigitizationModel(signal_length=1000, num_leads=12)
    y_adv = model_advanced(x)
    print(f"Advanced Model - Input shape: {x.shape}, Output shape: {y_adv.shape}")
    
    # Count parameters
    params_adv = sum(p.numel() for p in model_advanced.parameters() if p.requires_grad)
    print(f"Advanced Model - Parameters: {params_adv:,}")
    
    print()
    
    # Test loss function
    criterion = ECGDigitizationLoss()
    true_signals = torch.randn(2, 12, 1000)
    loss, loss_dict = criterion(y_adv, true_signals)
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss components: {loss_dict}")
    
    print("=" * 70)
    print("✓ All tests passed!")
