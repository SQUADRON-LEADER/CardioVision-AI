"""
Production-Ready ECG Image Digitization Flask Backend
Provides RESTful APIs for ECG image to signal conversion
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
import logging
from datetime import datetime
import traceback
from pathlib import Path
import json
import numpy as np

from model_loader import ModelManager
from inference import ECGInferenceEngine
from preprocessing import ECGImagePreprocessor
from config import Config
from flask.json.provider import DefaultJSONProvider

# Dummy class for unpickling notebook-saved models
class ECGDigitizationConfig:
    """Dummy config class for unpickling training checkpoint"""
    pass

# Custom JSON provider to handle NumPy types
class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
app.json = NumpyJSONProvider(app)
CORS(app)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ecg_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize components
model_manager = ModelManager(app.config['MODEL_PATH'])
preprocessor = ECGImagePreprocessor(target_size=app.config['IMAGE_SIZE'])
inference_engine = ECGInferenceEngine(model_manager, preprocessor)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load model at startup (before handling any requests)
logger.info("Starting ECG Digitization Service...")
logger.info(f"Model path: {app.config['MODEL_PATH']}")
logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
try:
    model_manager.load_model()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    logger.warning("Service starting without loaded model - requests will fail")


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def create_response(success, data=None, error=None, status_code=200):
    """Standardized API response format"""
    response = {
        'success': success,
        'timestamp': datetime.utcnow().isoformat(),
    }
    if data:
        response['data'] = data
    if error:
        response['error'] = error
    return jsonify(response), status_code


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        model_status = model_manager.is_loaded()
        return create_response(
            success=True,
            data={
                'status': 'healthy',
                'model_loaded': model_status,
                'model_version': model_manager.get_model_version(),
                'service': 'ECG Digitization Service',
                'version': '1.0.0'
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return create_response(
            success=False,
            error='Service unhealthy',
            status_code=503
        )


@app.route('/api/v1/info', methods=['GET'])
def get_service_info():
    """Get service information and capabilities"""
    info = {
        'service_name': 'ECG Image Digitization Service',
        'version': '1.0.0',
        'description': 'Converts ECG images to digital time-series signals',
        'capabilities': {
            'supported_formats': list(app.config['ALLOWED_EXTENSIONS']),
            'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024),
            'output_leads': 12,
            'signal_length': 1000,
            'sampling_rate_hz': 500,
            'signal_duration_sec': 2.0
        },
        'model_info': {
            'architecture': 'CNN-LSTM',
            'version': model_manager.get_model_version(),
            'parameters': model_manager.get_model_parameters(),
            'trained_on': '84 ECG samples with various conditions'
        },
        'preprocessing': {
            'techniques': [
                'Grid line removal',
                'Noise reduction',
                'Skew correction',
                'Rotation correction',
                'Illumination normalization',
                'Adaptive thresholding'
            ],
            'image_size': '256x256',
            'normalization': 'ImageNet mean/std'
        }
    }
    return create_response(success=True, data=info)


@app.route('/api/v1/digitize', methods=['POST'])
def digitize_ecg():
    """
    Main endpoint for ECG image digitization
    
    Request:
        - file: ECG image file (jpg, jpeg, png)
        - options: JSON object with processing options (optional)
            - remove_grid: bool (default: True)
            - denoise: bool (default: True)
            - correct_rotation: bool (default: True)
            - output_format: str ['json', 'csv', 'wfdb'] (default: 'json')
    
    Response:
        - success: bool
        - data: {
            - request_id: str
            - signals: dict of 12-lead ECG signals
            - metadata: processing metadata
            - quality_metrics: signal quality indicators
          }
    """
    try:
        # Validate request
        if 'file' not in request.files:
            return create_response(
                success=False,
                error='No file provided',
                status_code=400
            )
        
        file = request.files['file']
        
        if file.filename == '':
            return create_response(
                success=False,
                error='Empty filename',
                status_code=400
            )
        
        if not allowed_file(file.filename):
            return create_response(
                success=False,
                error=f'Invalid file type. Allowed: {app.config["ALLOWED_EXTENSIONS"]}',
                status_code=400
            )
        
        # Parse options
        options = {}
        if 'options' in request.form:
            import json
            options = json.loads(request.form['options'])
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_filename = f"{request_id}_{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(filepath)
        
        logger.info(f"Processing request {request_id} for file: {filename}")
        
        # Run inference
        result = inference_engine.process_ecg_image(
            image_path=filepath,
            request_id=request_id,
            options=options
        )
        
        # Save output if requested
        output_format = options.get('output_format', 'json')
        if output_format != 'json':
            output_path = inference_engine.save_output(
                result, 
                format=output_format,
                output_dir=app.config['OUTPUT_FOLDER']
            )
            result['output_file'] = output_path
        
        logger.info(f"Successfully processed request {request_id}")
        
        return create_response(
            success=True,
            data=result
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return create_response(
            success=False,
            error=str(e),
            status_code=400
        )
    except Exception as e:
        logger.error(f"Processing error: {str(e)}\n{traceback.format_exc()}")
        return create_response(
            success=False,
            error='Internal processing error',
            status_code=500
        )


@app.route('/api/v1/batch', methods=['POST'])
def batch_digitize():
    """
    Batch processing endpoint for multiple ECG images
    
    Request:
        - files: Multiple ECG image files
        - options: JSON object with processing options
    
    Response:
        - batch_id: str
        - results: list of individual results
        - summary: batch processing summary
    """
    try:
        if 'files' not in request.files:
            return create_response(
                success=False,
                error='No files provided',
                status_code=400
            )
        
        files = request.files.getlist('files')
        
        if len(files) == 0:
            return create_response(
                success=False,
                error='Empty file list',
                status_code=400
            )
        
        if len(files) > app.config['MAX_BATCH_SIZE']:
            return create_response(
                success=False,
                error=f'Batch size exceeds limit of {app.config["MAX_BATCH_SIZE"]}',
                status_code=400
            )
        
        # Parse options
        options = {}
        if 'options' in request.form:
            import json
            options = json.loads(request.form['options'])
        
        batch_id = str(uuid.uuid4())
        results = []
        successful = 0
        failed = 0
        
        logger.info(f"Processing batch {batch_id} with {len(files)} files")
        
        for file in files:
            try:
                if file and allowed_file(file.filename):
                    # Save file
                    filename = secure_filename(file.filename)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    saved_filename = f"{batch_id}_{timestamp}_{filename}"
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
                    file.save(filepath)
                    
                    # Process
                    result = inference_engine.process_ecg_image(
                        image_path=filepath,
                        request_id=f"{batch_id}_{filename}",
                        options=options
                    )
                    results.append({
                        'filename': filename,
                        'success': True,
                        'result': result
                    })
                    successful += 1
                else:
                    results.append({
                        'filename': file.filename,
                        'success': False,
                        'error': 'Invalid file type'
                    })
                    failed += 1
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}")
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': str(e)
                })
                failed += 1
        
        return create_response(
            success=True,
            data={
                'batch_id': batch_id,
                'total_files': len(files),
                'successful': successful,
                'failed': failed,
                'results': results
            }
        )
        
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        return create_response(
            success=False,
            error='Batch processing failed',
            status_code=500
        )


@app.route('/api/v1/export/<format>', methods=['POST'])
def export_signals(format):
    """
    Export digitized signals in various formats
    
    Parameters:
        format: 'csv', 'json', 'wfdb', 'matlab'
    
    Request:
        - signals: JSON object containing 12-lead signals
        - metadata: Optional metadata
    
    Response:
        - File download or JSON with download link
    """
    try:
        if format not in ['csv', 'json', 'wfdb', 'matlab']:
            return create_response(
                success=False,
                error=f'Unsupported format: {format}',
                status_code=400
            )
        
        data = request.get_json()
        if not data or 'signals' not in data:
            return create_response(
                success=False,
                error='No signal data provided',
                status_code=400
            )
        
        # Export signals
        output_path = inference_engine.export_signals(
            signals=data['signals'],
            metadata=data.get('metadata', {}),
            format=format,
            output_dir=app.config['OUTPUT_FOLDER']
        )
        
        # Return file for download
        return send_file(
            output_path,
            as_attachment=True,
            download_name=os.path.basename(output_path)
        )
        
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        return create_response(
            success=False,
            error='Export failed',
            status_code=500
        )


@app.route('/api/v1/validate', methods=['POST'])
def validate_image():
    """
    Validate ECG image quality before processing
    
    Request:
        - file: ECG image file
    
    Response:
        - valid: bool
        - quality_score: float (0-1)
        - issues: list of detected issues
        - recommendations: list of recommendations
    """
    try:
        if 'file' not in request.files:
            return create_response(
                success=False,
                error='No file provided',
                status_code=400
            )
        
        file = request.files['file']
        
        # Save temporarily
        temp_path = os.path.join(
            app.config['UPLOAD_FOLDER'],
            f"temp_{uuid.uuid4()}_{secure_filename(file.filename)}"
        )
        file.save(temp_path)
        
        try:
            # Validate image
            validation_result = preprocessor.validate_image(temp_path)
            
            return create_response(
                success=True,
                data=validation_result
            )
        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return create_response(
            success=False,
            error='Validation failed',
            status_code=500
        )


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return create_response(
        success=False,
        error='File too large. Maximum size: {} MB'.format(
            app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
        ),
        status_code=413
    )


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return create_response(
        success=False,
        error='Endpoint not found',
        status_code=404
    )


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    logger.error(f"Internal error: {str(e)}")
    return create_response(
        success=False,
        error='Internal server error',
        status_code=500
    )


if __name__ == '__main__':
    # Run Flask app
    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        debug=app.config['DEBUG']
    )
