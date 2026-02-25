# CardioVision AI - Backend & Frontend Integration Guide

## ✅ Integration Complete!

The newly trained HybridECGNet model has been successfully integrated with both the backend and frontend.

## 🎯 Model Information

- **Model Name**: HybridECGNet
- **Architecture**: Hybrid CNN with Residual Blocks + CBAM Attention
- **Task**: ECG Image Classification
- **Classes**: 
  1. Normal
  2. Abnormal Heartbeat
  3. Myocardial Infarction
- **Test Accuracy**: ~70% (from final training)
- **Model Location**: `flask_backend/ecg_model_final.pth`

## 🔧 Changes Made

### 1. Backend Updates (`flask_backend/`)

#### `model_loader.py`
- ✅ Added HybridECGNet architecture classes:
  - `ChannelAttention`
  - `SpatialAttention`
  - `CBAM` (Convolutional Block Attention Module)
  - `ResidualBlock`
  - `HybridECGNet`
- ✅ Updated `ModelManager.load_model()` to auto-detect model architecture
- ✅ Added support for classification metadata

#### `inference.py`
- ✅ Added `CLASS_NAMES` for classification
- ✅ Updated `__init__` to detect model type
- ✅ Modified `process_ecg_image()` to handle both classification and digitization
- ✅ Added `_post_process_classification()` method for classification results

#### `config.py`
- ✅ Updated `MODEL_PATH` to point to new model: `flask_backend/ecg_model_final.pth`
- ✅ Updated `MODEL_VERSION` to `2.0.0`
- ✅ Updated `IMAGE_SIZE` from `(256, 256)` to `(224, 224)` for HybridECGNet

### 2. Frontend Updates (`frontend/`)

#### `index.html`
- ✅ Updated hero title: "Analyze ECG Images with AI Precision"
- ✅ Changed "Digitization Results" to "Analysis Results"
- ✅ Updated card labels: "Quality Score" → "Confidence", "Leads Extracted" → "Classification"
- ✅ Modified canvas size for better visualization

#### `app.js`
- ✅ Added `displayClassificationResults()` function
- ✅ Added `drawClassificationChart()` for visualization
- ✅ Updated `displayResults()` to handle both classification and digitization
- ✅ Modified CSV download to support classification results
- ✅ Added probability distribution visualization

## 🚀 How to Run

### Start Backend Server

```powershell
cd flask_backend
python app.py
```

The server will start on `http://localhost:5000`

**Expected Output:**
```
INFO - ModelManager initialized with device: cpu
INFO - Detected model architecture: HybridECGNet
INFO - Number of classes: 3
INFO - Initializing HybridECGNet...
INFO - Model loaded successfully
* Running on http://127.0.0.1:5000
```

### Start Frontend

1. Open `frontend/index.html` in your web browser
2. Or use a simple HTTP server:

```powershell
cd frontend
python -m http.server 8000
```

Then navigate to `http://localhost:8000`

## 📊 API Usage

### Classification Endpoint

```bash
POST http://localhost:5000/api/v1/digitize
Content-Type: multipart/form-data

file: <ECG_IMAGE_FILE>
options: {"remove_grid": true, "denoise": true}
```

### Response Format (Classification)

```json
{
  "success": true,
  "data": {
    "request_id": "uuid",
    "prediction": {
      "class": "Normal",
      "class_index": 0,
      "confidence": 85.5,
      "probability_distribution": {
        "Normal": 0.855,
        "Abnormal Heartbeat": 0.120,
        "Myocardial Infarction": 0.025
      }
    },
    "metadata": {
      "task": "classification",
      "model_type": "HybridECGNet",
      "processing_time_seconds": 0.234,
      "num_classes": 3
    },
    "quality_metrics": {
      "confidence_score": 85.5,
      "prediction_certainty": "high",
      "entropy": 0.456
    }
  }
}
```

## 🎨 Frontend Features

### Classification View
- **Predicted Class**: Displayed prominently with color coding:
  - 🟢 Normal (Green)
  - 🟠 Abnormal Heartbeat (Orange)
  - 🔴 Myocardial Infarction (Red)
- **Confidence Score**: Percentage confidence in prediction
- **Probability Distribution**: Bar chart showing probabilities for all classes

### Download Options
- **JSON**: Complete result data in JSON format
- **CSV**: Classification results in CSV format with probabilities

## 🧪 Testing

### Test with Sample Images

1. Use ECG images from `ecg_data/` directories:
   - `ecg_data/normal_ecg_images/`
   - `ecg_data/abnormal_heartbeat_ecg_images/`
   - `ecg_data/myocardial_infarction_ecg_images/`

2. Upload an image through the frontend interface
3. Click "Process ECG Image"
4. View classification results with confidence scores

### API Testing with cURL

```bash
curl -X POST http://localhost:5000/api/v1/digitize \
  -F "file=@path/to/ecg_image.png" \
  -F 'options={"remove_grid":true,"denoise":true}'
```

## 📁 File Structure

```
CardioVision-AI/
├── flask_backend/
│   ├── app.py                    # Main Flask application
│   ├── model_loader.py           # Model loading with HybridECGNet ✅
│   ├── inference.py              # Inference engine ✅
│   ├── preprocessing.py          # Image preprocessing
│   ├── config.py                 # Configuration ✅
│   ├── ecg_model_final.pth       # New trained model ✅
│   └── best_model_advanced.pth   # Best checkpoint ✅
├── frontend/
│   ├── index.html                # Main UI ✅
│   ├── app.js                    # Frontend logic ✅
│   └── styles.css                # Styles
├── outputs/
│   └── ecg_digitization_model_final.pth  # Original model
├── checkpoints/
│   └── best_model_advanced.pth   # Training checkpoint
└── INTEGRATION_GUIDE.md          # This file

✅ = Updated for new model
```

## 🔍 Troubleshooting

### Model Loading Issues
- Ensure `ecg_model_final.pth` exists in `flask_backend/`
- Check Python console for detailed error messages
- Verify PyTorch is installed: `pip install torch torchvision`

### Frontend Connection Issues
- Verify backend is running on port 5000
- Check browser console for CORS or network errors
- Ensure API_URL in `app.js` matches backend URL

### Image Upload Issues
- Supported formats: JPG, PNG
- Maximum file size: 16MB
- Image should contain ECG traces

## 🎉 Success Indicators

When everything is working correctly, you should see:

1. ✅ Backend starts without errors
2. ✅ Model loads as "HybridECGNet" architecture
3. ✅ Frontend displays upload interface
4. ✅ Image upload and processing succeeds
5. ✅ Classification results display with confidence scores
6. ✅ Probability distribution bars show for all 3 classes

## 📝 Next Steps

1. **Improve Model Accuracy**: Current model is at ~70%, can be retrained for higher accuracy
2. **Add More Classes**: Expand classification to more ECG conditions
3. **Production Deployment**: Configure for production with gunicorn, nginx
4. **Authentication**: Add user authentication and API keys
5. **Batch Processing**: Support multiple image uploads

## 💡 Tips

- Use clear, high-quality ECG images for best results
- The model performs better on properly aligned ECG traces
- Confidence scores below 50% indicate uncertain predictions
- Download JSON for complete metadata and debugging

---

**Last Updated**: February 25, 2026
**Model Version**: 2.0.0 (HybridECGNet)
**Status**: ✅ Fully Integrated and Operational
