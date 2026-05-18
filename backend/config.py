"""
Configuration for ECG Digitization Service
"""

import os
from pathlib import Path


def _first_existing_path(candidates):
    """Return first existing path from candidates, otherwise first candidate."""
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return str(candidates[0]) if candidates else ''

class Config:
    """Flask application configuration"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'ecg-digitization-secret-key-change-in-production'
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))
    
    # File upload settings
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    OUTPUT_FOLDER = os.environ.get('OUTPUT_FOLDER', 'outputs')
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB default
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'}
    
    # Model settings
    BASE_DIR = Path(__file__).parent.parent

    _CLASSIFICATION_CANDIDATES = [
        BASE_DIR / 'backend' / 'ecg_model_final.pth',
        BASE_DIR / 'backend' / 'best_model_advanced.pth',
    ]
    _DIGITIZATION_CANDIDATES = [
        BASE_DIR / 'outputs' / 'ecg_digitization_effnet_final.pth',
        BASE_DIR / 'outputs' / 'ecg_digitization_model_final.pth',
        BASE_DIR / 'backend' / 'best_model_ptbxl.pth',
    ]
    
    # Support both classification and digitization models
    MODEL_PATHS = {
        'classification': os.environ.get(
            'CLASSIFICATION_MODEL',
            _first_existing_path(_CLASSIFICATION_CANDIDATES),
        ),
        'digitization': os.environ.get(
            'DIGITIZATION_MODEL',
            _first_existing_path(_DIGITIZATION_CANDIDATES),
        ),
    }
    
    # Legacy single model path (for backward compatibility)
    MODEL_PATH = MODEL_PATHS['classification']
    MODEL_VERSION = '3.0.0'  # Updated to support both tasks
    
    # Default task (what to use if not specified)
    # Supported: 'classification', 'digitization', 'pipeline', 'auto'
    DEFAULT_TASK = os.environ.get('DEFAULT_TASK', 'pipeline')
    
    # Processing settings
    MAX_BATCH_SIZE = int(os.environ.get('MAX_BATCH_SIZE', 10))
    DEVICE = os.environ.get('DEVICE', 'cpu')  # 'cpu' or 'cuda'
    
    # ECG signal parameters
    SIGNAL_LENGTH = 1000
    NUM_LEADS = 12
    SAMPLING_RATE = 500  # Hz
    SIGNAL_DURATION = 2.0  # seconds
    
    LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # Image preprocessing
    IMAGE_SIZE = (224, 224)  # Updated for HybridECGNet
    NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
    NORMALIZATION_STD = [0.229, 0.224, 0.225]
    
    # Quality thresholds
    MIN_IMAGE_WIDTH = 200
    MIN_IMAGE_HEIGHT = 200
    MIN_QUALITY_SCORE = 0.3
    
    # CORS settings
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*')
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'ecg_service.log')
    
    # Cache settings (for future optimization)
    ENABLE_CACHE = os.environ.get('ENABLE_CACHE', 'False').lower() == 'true'
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300


def get_config_dict():
    """Return a dictionary of key service configuration values.

    Useful for CLI tools, tests, and runtime inspection without instantiating
    the Flask app object.
    """
    keys = [
        'HOST', 'PORT', 'UPLOAD_FOLDER', 'OUTPUT_FOLDER', 'MODEL_PATHS',
        'DEFAULT_TASK', 'IMAGE_SIZE', 'LOG_LEVEL', 'LOG_FILE', 'DEVICE',
        'MAX_BATCH_SIZE'
    ]
    cfg = {}
    for k in keys:
        cfg[k] = getattr(Config, k, None)
    return cfg
