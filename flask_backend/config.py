"""
Configuration for ECG Digitization Service
"""

import os
from pathlib import Path

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
    MODEL_PATH = os.environ.get('MODEL_PATH', str(BASE_DIR / 'checkpoints' / 'best_model.pth'))
    MODEL_VERSION = '1.0.0'
    
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
    IMAGE_SIZE = (256, 256)
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
