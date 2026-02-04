# ECG Image Digitization Service - Flask Backend

Production-ready Flask backend for converting ECG images to digital time-series signals using deep learning.

## Features

- **RESTful API**: Clean, documented endpoints for ECG digitization
- **Robust Preprocessing**: Handles grid lines, noise, skew, rotation, illumination
- **Production Ready**: Error handling, logging, validation, CORS
- **Multiple Export Formats**: JSON, CSV, WFDB (PhysioNet compatible)
- **Batch Processing**: Process multiple ECG images efficiently
- **Quality Metrics**: Comprehensive signal quality indicators
- **Model Versioning**: Track model versions and metadata

## Architecture

```
flask_backend/
├── app.py                 # Main Flask application
├── config.py              # Configuration settings
├── model_loader.py        # Model loading and management
├── preprocessing.py       # Image preprocessing pipeline
├── inference.py          # Inference orchestration
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file:

```bash
# Model Configuration
MODEL_PATH=../checkpoints/best_model.pth
DEVICE=cpu  # or 'cuda' for GPU

# Server Configuration
HOST=0.0.0.0
PORT=5000
DEBUG=False

# Upload Configuration
UPLOAD_FOLDER=uploads
OUTPUT_FOLDER=outputs
MAX_CONTENT_LENGTH=16777216  # 16MB

# Processing
MAX_BATCH_SIZE=10
```

### 3. Run Server

```bash
python app.py
```

Or use Gunicorn for production:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## API Documentation

### 1. Health Check

**Endpoint**: `GET /health`

**Response**:
```json
{
  "success": true,
  "timestamp": "2024-01-22T12:00:00",
  "data": {
    "status": "healthy",
    "model_loaded": true,
    "model_version": "1.0.0",
    "service": "ECG Digitization Service"
  }
}
```

### 2. Service Information

**Endpoint**: `GET /api/v1/info`

**Response**:
```json
{
  "success": true,
  "data": {
    "service_name": "ECG Image Digitization Service",
    "version": "1.0.0",
    "capabilities": {
      "supported_formats": ["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
      "max_file_size_mb": 16,
      "output_leads": 12,
      "signal_length": 1000,
      "sampling_rate_hz": 500
    },
    "model_info": {
      "architecture": "CNN-LSTM",
      "version": "1.0.0"
    }
  }
}
```

### 3. Digitize ECG Image

**Endpoint**: `POST /api/v1/digitize`

**Request**:
- Content-Type: `multipart/form-data`
- Body:
  - `file`: ECG image file
  - `options` (optional): JSON string with processing options

```bash
curl -X POST http://localhost:5000/api/v1/digitize \
  -F "file=@ecg_image.jpg" \
  -F 'options={"remove_grid":true,"denoise":true,"output_format":"json"}'
```

**Response**:
```json
{
  "success": true,
  "timestamp": "2024-01-22T12:00:00",
  "data": {
    "request_id": "uuid-here",
    "signals": {
      "I": [0.1, 0.2, ...],
      "II": [0.3, 0.4, ...],
      ...
      "V6": [0.5, 0.6, ...]
    },
    "metadata": {
      "processing_time_seconds": 0.856,
      "sampling_rate_hz": 500,
      "signal_duration_sec": 2.0,
      "num_leads": 12,
      "signal_length": 1000
    },
    "quality_metrics": {
      "quality_score": 0.85,
      "overall": {
        "mean_snr_db": 25.3
      },
      "per_lead": { ... }
    }
  }
}
```

### 4. Batch Processing

**Endpoint**: `POST /api/v1/batch`

**Request**:
```bash
curl -X POST http://localhost:5000/api/v1/batch \
  -F "files=@ecg1.jpg" \
  -F "files=@ecg2.jpg" \
  -F "files=@ecg3.jpg"
```

**Response**:
```json
{
  "success": true,
  "data": {
    "batch_id": "batch-uuid",
    "total_files": 3,
    "successful": 3,
    "failed": 0,
    "results": [...]
  }
}
```

### 5. Validate Image

**Endpoint**: `POST /api/v1/validate`

**Request**:
```bash
curl -X POST http://localhost:5000/api/v1/validate \
  -F "file=@ecg_image.jpg"
```

**Response**:
```json
{
  "success": true,
  "data": {
    "valid": true,
    "quality_score": 0.85,
    "dimensions": {"width": 1200, "height": 800},
    "mean_brightness": 180.5,
    "issues": [],
    "recommendations": []
  }
}
```

### 6. Export Signals

**Endpoint**: `POST /api/v1/export/<format>`

**Formats**: `csv`, `json`, `wfdb`, `matlab`

**Request**:
```bash
curl -X POST http://localhost:5000/api/v1/export/csv \
  -H "Content-Type: application/json" \
  -d '{"signals": {...}, "metadata": {...}}' \
  --output signals.csv
```

## Processing Options

Configure preprocessing behavior:

```json
{
  "remove_grid": true,        // Remove ECG grid lines
  "denoise": true,            // Apply denoising
  "correct_rotation": true,   // Auto-rotate image
  "correct_skew": true,       // Correct perspective skew
  "output_format": "json"     // Output format: json, csv, wfdb
}
```

## Signal Output Format

### 12-Lead ECG
- **Leads**: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
- **Sampling Rate**: 500 Hz
- **Duration**: 2.0 seconds
- **Samples per Lead**: 1000
- **Units**: Normalized amplitude

### Quality Metrics
- **quality_score**: Overall signal quality (0-1)
- **mean_snr_db**: Signal-to-noise ratio
- **per_lead**: Individual lead statistics

## Error Handling

All endpoints return standardized error responses:

```json
{
  "success": false,
  "timestamp": "2024-01-22T12:00:00",
  "error": "Error description"
}
```

**HTTP Status Codes**:
- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Endpoint not found
- `413`: File too large
- `500`: Internal server error
- `503`: Service unavailable

## Deployment

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p uploads outputs

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

Build and run:

```bash
docker build -t ecg-digitization-service .
docker run -p 5000:5000 -v /path/to/model:/app/checkpoints ecg-digitization-service
```

### Production Considerations

1. **Use HTTPS**: Deploy behind reverse proxy (nginx, Apache)
2. **Authentication**: Add API key or OAuth2
3. **Rate Limiting**: Implement request throttling
4. **Monitoring**: Add metrics and health checks
5. **Scaling**: Use load balancer for multiple instances
6. **Model Caching**: Cache model in memory
7. **Async Processing**: Use Celery for long-running tasks

## Performance Optimization

- **Batch Processing**: Process multiple images efficiently
- **Model Warmup**: Pre-load model at startup
- **Image Caching**: Cache preprocessed images
- **GPU Acceleration**: Use CUDA for faster inference
- **Connection Pooling**: Optimize database connections

## Testing

Run tests:

```bash
pytest tests/
```

Test coverage:

```bash
pytest --cov=. tests/
```

## Monitoring

View logs:

```bash
tail -f ecg_service.log
```

Metrics to monitor:
- Request rate
- Processing time
- Error rate
- Model inference time
- Memory usage
- CPU/GPU utilization

## Troubleshooting

### Model Loading Fails
- Check MODEL_PATH configuration
- Verify checkpoint file exists
- Ensure model architecture matches

### Low Quality Scores
- Validate input image quality
- Check preprocessing options
- Review signal characteristics

### Slow Performance
- Enable GPU (DEVICE=cuda)
- Reduce image size
- Use batch processing
- Optimize preprocessing

## Support

For issues and questions:
- GitHub Issues
- Documentation
- Email support

## License

MIT License - see LICENSE file for details
