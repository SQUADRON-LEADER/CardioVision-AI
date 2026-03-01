"""
Updated Model Loader for ECG Digitization
Supports both classification and digitization models
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages loading and inference for ECG models
    Supports both classification and digitization models
    """
    
    def __init__(self, model_paths):
        """
        Args:
            model_paths: dict with 'classification' and 'digitization' keys
                        or single path string for backward compatibility
        """
        if isinstance(model_paths, str):
            # Backward compatibility - single path
            self.model_paths = {'classification': model_paths}
        else:
            self.model_paths = model_paths
        
        self.models = {}
        self.model_info = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Model Manager initialized on device: {self.device}")
    
    def load_model(self, model_type='auto'):
        """
        Load a specific model type
        
        Args:
            model_type: 'classification', 'digitization', or 'auto'
        """
        if model_type == 'auto':
            # Load all available models
            for mtype in self.model_paths.keys():
                self._load_specific_model(mtype)
        else:
            self._load_specific_model(model_type)
    
    def _load_specific_model(self, model_type):
        """Load a specific model"""
        if model_type not in self.model_paths:
            logger.warning(f"Model type '{model_type}' not configured")
            return
        
        model_path = self.model_paths[model_type]
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            return
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Determine model architecture and load
            if model_type == 'classification':
                model = self._load_classification_model(checkpoint)
            elif model_type == 'digitization':
                model = self._load_digitization_model(checkpoint)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            self.models[model_type] = model
            
            # Store metadata
            self.model_info[model_type] = {
                'path': model_path,
                'type': model_type,
                'parameters': sum(p.numel() for p in model.parameters()),
                'device': str(self.device)
            }
            
            logger.info(f"✓ {model_type.capitalize()} model loaded successfully")
            logger.info(f"  Parameters: {self.model_info[model_type]['parameters']:,}")
            
        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {str(e)}")
            raise
    
    def _load_classification_model(self, checkpoint):
        """Load classification model (HybridECGNet)"""
        from flask_backend.model_loader import HybridECGNet
        
        model = HybridECGNet(num_classes=3, dropout=0.3)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def _load_digitization_model(self, checkpoint):
        """Load digitization model"""
        # Import from parent directory
        try:
            from ECG_Digitization_Model import create_digitization_model
        except ImportError:
            # Try alternative import
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from ECG_Digitization_Model import create_digitization_model
        
        # Detect model type from checkpoint
        model_type = 'advanced'  # default
        if 'model_type' in checkpoint:
            model_type = checkpoint['model_type']
        
        model = create_digitization_model(
            model_type=model_type,
            signal_length=1000,
            num_leads=12,
            dropout=0.3
        )
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def predict(self, input_tensor, model_type='auto'):
        """
        Run inference
        
        Args:
            input_tensor: Input image tensor (B, 3, H, W)
            model_type: Which model to use ('classification', 'digitization', 'auto')
        
        Returns:
            predictions: Model output
        """
        # Auto-detect model type if not specified
        if model_type == 'auto':
            if 'digitization' in self.models:
                model_type = 'digitization'
            elif 'classification' in self.models:
                model_type = 'classification'
            else:
                raise ValueError("No models loaded")
        
        if model_type not in self.models:
            raise ValueError(f"Model type '{model_type}' not loaded")
        
        model = self.models[model_type]
        
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            predictions = model(input_tensor)
        
        return predictions, model_type
    
    def is_loaded(self, model_type=None):
        """Check if model(s) are loaded"""
        if model_type:
            return model_type in self.models
        return len(self.models) > 0
    
    def get_model_info(self, model_type=None):
        """Get model information"""
        if model_type:
            return self.model_info.get(model_type, {})
        return self.model_info
    
    def get_model_version(self):
        """Get model version string"""
        loaded_types = list(self.models.keys())
        if not loaded_types:
            return "No models loaded"
        return f"Models: {', '.join(loaded_types)}"
    
    def get_model_parameters(self):
        """Get total model parameters"""
        total = 0
        for model in self.models.values():
            total += sum(p.numel() for p in model.parameters())
        return total
    
    def get_available_models(self):
        """Get list of available model types"""
        return list(self.models.keys())


# Legacy HybridECGNet for classification (keep for backward compatibility)
class HybridECGNet(nn.Module):
    """
    Classification model for ECG images
    """
    
    def __init__(self, num_classes=3, dropout=0.3):
        super(HybridECGNet, self).__init__()
        
        import torchvision.models as models
        
        # Encoder: ResNet34 backbone
        resnet = models.resnet34(pretrained=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Extract features
        features = self.encoder(x)  # (B, 512, H, W)
        
        # Channel attention
        ca = self.channel_attention(features)  # (B, 512)
        ca = ca.unsqueeze(-1).unsqueeze(-1)  # (B, 512, 1, 1)
        features = features * ca
        
        # Spatial attention
        sa = self.spatial_attention(features)  # (B, 1, H, W)
        features = features * sa
        
        # Classify
        output = self.classifier(features)  # (B, num_classes)
        
        return output


if __name__ == "__main__":
    # Test model manager
    print("Testing Model Manager...")
    print("=" * 70)
    
    # Test with dummy paths
    manager = ModelManager({
        'classification': 'best_model_advanced.pth',
        'digitization': 'best_digitization_model.pth'
    })
    
    print(f"Available models: {manager.get_available_models()}")
    print("=" * 70)
