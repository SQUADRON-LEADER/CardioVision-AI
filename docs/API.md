# API Documentation

## Endpoints

### Health Check
**GET** `/health`
- Returns service health and model status

### Service Info
**GET** `/api/v1/info`
- Returns service capabilities and model information

### Process ECG
**POST** `/api/v1/digitize`
- Process single ECG image
- Parameters:
  - file: Image file
  - task: 'classification', 'digitization', or 'pipeline'
  - options: JSON with processing options

Response:
```json
{
  "success": true,
  "data": {
    "request_id": "uuid",
    "task": "digitization",
    "signals": {...},
    "metadata": {...}
  }
}
```

### Batch Processing
**POST** `/api/v1/batch`
- Process multiple ECG images
- Parameters: files, options
- Returns batch results with summary

### Export Signals
**POST** `/api/v1/export/<format>`
- Export signals in various formats
- Formats: json, csv, wfdb, matlab
