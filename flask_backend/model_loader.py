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
            
            # Create dummy config class that might be in the checkpoint
            class ECGDigitizationConfig:
                pass
            
            # Add to __main__ module namespace for unpickling
            sys.modules['__main__'].ECGDigitizationConfig = ECGDigitizationConfig
            
            # Load checkpoint with weights_only=False to handle notebook-saved models
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            except Exception as e:
                logger.error(f"Failed to load model checkpoint: {e}")
                raise
            
            # Initialize digitization model
            self.model = ECGDigitizationModel(
                num_leads=12,
                signal_length=1000,
                lstm_hidden_dim=128,
                lstm_layers=1
            )
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model_metadata = {
                    'epoch': checkpoint.get('epoch', 'unknown'),
                    'val_loss': checkpoint.get('val_loss', 'unknown'),
                    'train_loss': checkpoint.get('train_loss', 'unknown'),
                }
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
        
        return {
            'status': 'loaded',
            'path': str(self.model_path),
            'device': str(self.device),
            'parameters': self.get_model_parameters(),
            'version': self.get_model_version(),
            'metadata': self.model_metadata,
            'architecture': {
                'type': 'CNN-LSTM Digitization',
                'input_size': '(3, 256, 256)',
                'output_size': '(12, 1000)',
                'num_leads': 12,
                'signal_length': 1000,
                'task': 'ECG Image to Signal Digitization',
                'backbone': 'SimpleCNN + BiLSTM + Attention'
            }
        }
    
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
