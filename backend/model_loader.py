"""
Model loading and management for ECG Digitization Service
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path
import json
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)


# ============================================================================
# NEW: HybridECGNet Architecture from Training Notebook
# ============================================================================

class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ResidualBlock(nn.Module):
    """Residual Block with Batch Normalization and Attention"""
    def __init__(self, in_channels, out_channels, stride=1, use_attention=True):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.attention = CBAM(out_channels) if use_attention else None
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.attention:
            out = self.attention(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class HybridECGNet(nn.Module):
    """Hybrid CNN for ECG Classification with Attention and Residual Connections"""
    
    def __init__(self, num_classes=3, dropout_rate=0.3):
        super(HybridECGNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks with increasing channels
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool_global = nn.AdaptiveMaxPool2d((1, 1))
        
        # Classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(512 * 2, 512)  # *2 for avg and max pool concatenation
        self.bn_fc = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, use_attention=True))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, use_attention=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling (both avg and max)
        x_avg = self.avgpool(x)
        x_max = self.maxpool_global(x)
        x = torch.cat([x_avg, x_max], dim=1)
        x = torch.flatten(x, 1)
        
        # Classifier
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# ============================================================================
# Original Models (Legacy Support)
# ============================================================================

class SimpleCNN(nn.Module):
    """CNN backbone for feature extraction"""
    
    def __init__(self, input_channels=3):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.features(x)


class ECGDigitizationModel(nn.Module):
    """
    ECG Digitization Model - Image to Signals
    Input: ECG image (3, 256, 256)
    Output: 12-lead signals (12, 1000)
    """
    
    def __init__(self, num_leads=12, signal_length=1000, lstm_hidden_dim=128, lstm_layers=1):
        super().__init__()
        self.num_leads = num_leads
        self.signal_length = signal_length
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        
        # Vision backbone
        self.backbone = SimpleCNN(input_channels=3)
        backbone_channels = 512
        
        # Feature compression
        self.feature_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_channels * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Sequence decoder - BiLSTM
        self.sequence_decoder = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if lstm_layers > 1 else 0
        )
        
        # Lead attention
        lstm_output_size = lstm_hidden_dim * 2
        self.lead_attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Final projection
        self.final_projection = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_leads)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract visual features
        visual_features = self.backbone(x)
        
        # Project to sequence format
        projected_features = self.feature_projection(visual_features)
        
        # Expand for sequence generation
        sequence_input = projected_features.unsqueeze(1).expand(
            batch_size, self.signal_length, -1
        )
        
        # Generate sequence with LSTM
        lstm_output, _ = self.sequence_decoder(sequence_input)
        
        # Apply attention
        attended_output, _ = self.lead_attention(lstm_output, lstm_output, lstm_output)
        
        # Generate multi-lead output
        output = self.final_projection(attended_output)
        
        # Transpose to [B, num_leads, signal_length]
        output = output.transpose(1, 2)
        
        return output


class ModelManager:
    """Manages model loading, versioning, and inference"""
    
    def __init__(self, model_path, device='cpu'):
        self.model_path = Path(model_path)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_metadata = {}
        self._loaded = False
        
        logger.info(f"ModelManager initialized with device: {self.device}")
    
    def load_model(self):
        """Load pre-trained model from checkpoint"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            logger.info(f"Loading model from: {self.model_path}")
            
            # Setup dummy classes for unpickling BEFORE loading
            import sys
            
            # Create dummy classes that might be in the checkpoint
            class ECGDigitizationConfig:
                pass
            
            # Add necessary classes to __main__ module namespace for unpickling
            sys.modules['__main__'].ECGDigitizationConfig = ECGDigitizationConfig
            sys.modules['__main__'].HybridECGNet = HybridECGNet
            sys.modules['__main__'].ChannelAttention = ChannelAttention
            sys.modules['__main__'].SpatialAttention = SpatialAttention
            sys.modules['__main__'].CBAM = CBAM
            sys.modules['__main__'].ResidualBlock = ResidualBlock
            
            # Load checkpoint with weights_only=False to handle notebook-saved models
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            except Exception as e:
                logger.error(f"Failed to load model checkpoint: {e}")
                raise
            
            # Detect model architecture and initialize appropriate model
            model_arch = checkpoint.get('model_architecture', 'ECGDigitizationModel')
            num_classes = checkpoint.get('num_classes', 3)
            
            logger.info(f"Detected model architecture: {model_arch}")
            logger.info(f"Number of classes: {num_classes}")
            
            if model_arch == 'HybridECGNet' or 'layer1' in str(checkpoint.get('model_state_dict', {}).keys()):
                # Load new HybridECGNet model
                logger.info("Initializing HybridECGNet...")
                self.model = HybridECGNet(num_classes=num_classes, dropout_rate=0.3)
                self.model_metadata = {
                    'architecture': 'HybridECGNet',
                    'num_classes': num_classes,
                    'test_accuracy': checkpoint.get('test_accuracy', 'unknown'),
                    'test_precision': checkpoint.get('test_precision', 'unknown'),
                    'test_recall': checkpoint.get('test_recall', 'unknown'),
                    'test_f1': checkpoint.get('test_f1', 'unknown'),
                    'best_val_accuracy': checkpoint.get('best_val_accuracy', 'unknown'),
                }
            else:
                # Legacy model - Initialize digitization model
                logger.info("Initializing ECGDigitizationModel (legacy)...")
                self.model = ECGDigitizationModel(
                    num_leads=12,
                    signal_length=1000,
                    lstm_hidden_dim=128,
                    lstm_layers=1
                )
                self.model_metadata = {
                    'architecture': 'ECGDigitizationModel',
                    'epoch': checkpoint.get('epoch', 'unknown'),
                    'val_loss': checkpoint.get('val_loss', 'unknown'),
                    'train_loss': checkpoint.get('train_loss', 'unknown'),
                }
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            self._loaded = True
            logger.info("Model loaded successfully")
            logger.info(f"Model metadata: {self.model_metadata}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def is_loaded(self):
        """Check if model is loaded"""
        return self._loaded
    
    def get_model(self):
        """Get the loaded model"""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model
    
    def get_device(self):
        """Get the device model is on"""
        return self.device
    
    def get_model_version(self):
        """Get model version"""
        return self.model_metadata.get('epoch', '1.0.0')
    
    def get_model_parameters(self):
        """Get model parameter count"""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())
    
    def get_model_info(self):
        """Get comprehensive model information"""
        if not self._loaded:
            return {'status': 'not_loaded'}
        
        arch = self.model_metadata.get('architecture', 'unknown')
        
        base_info = {
            'status': 'loaded',
            'path': str(self.model_path),
            'device': str(self.device),
            'parameters': self.get_model_parameters(),
            'version': self.get_model_version(),
            'metadata': self.model_metadata,
        }
        
        if arch == 'HybridECGNet':
            base_info['architecture'] = {
                'type': 'HybridECGNet - Advanced Classification',
                'input_size': '(3, 224, 224)',
                'output_size': f'({self.model_metadata.get("num_classes", 3)} classes)',
                'num_classes': self.model_metadata.get('num_classes', 3),
                'class_names': ['Normal', 'Abnormal Heartbeat', 'Myocardial Infarction'],
                'task': 'ECG Image Classification',
                'features': [
                    'Residual Connections',
                    'Channel & Spatial Attention (CBAM)',
                    'Multi-scale Feature Extraction',
                    'Batch Normalization & Dropout'
                ],
                'test_accuracy': self.model_metadata.get('test_accuracy', 'N/A'),
                'test_f1': self.model_metadata.get('test_f1', 'N/A')
            }
        else:
            base_info['architecture'] = {
                'type': 'CNN-LSTM Digitization',
                'input_size': '(3, 256, 256)',
                'output_size': '(12, 1000)',
                'num_leads': 12,
                'signal_length': 1000,
                'task': 'ECG Image to Signal Digitization',
                'backbone': 'SimpleCNN + BiLSTM + Attention'
            }
        
        return base_info
    
    @torch.no_grad()
    def predict(self, input_tensor):
        """
        Run inference on input tensor
        
        Args:
            input_tensor: torch.Tensor of shape (B, 3, 256, 256)
        
        Returns:
            predictions: torch.Tensor of shape (B, 12, 1000)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # Move input to device
        input_tensor = input_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        return predictions
    
    def warmup(self, num_iterations=5):
        """Warmup model for faster inference"""
        logger.info("Warming up model...")
        
        dummy_input = torch.randn(1, 3, 256, 256).to(self.device)
        
        for i in range(num_iterations):
            _ = self.predict(dummy_input)
        
        logger.info("Model warmup complete")
