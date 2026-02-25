# CardioVision-AI

An advanced ECG image classification system using deep learning with HybridECGNet architecture for automated heart condition detection.

## 🎯 Overview

CardioVision-AI is a comprehensive electrocardiogram (ECG) analysis platform that leverages state-of-the-art hybrid CNN architecture with attention mechanisms to classify cardiac conditions:

- ✅ **Normal ECG patterns**
- ✅ **Abnormal heartbeat patterns**
- ✅ **Myocardial Infarction (Heart Attack)**

## ✨ Features

- **🤖 Advanced AI Model**: HybridECGNet with Residual Blocks + CBAM Attention
- **📊 Real-time Classification**: Upload ECG images and receive instant predictions
- **🎨 Modern Web Interface**: Beautiful, responsive frontend with visualization
- **🔌 REST API**: Flask backend for easy integration
- **📈 High Performance**: Trained on clinical-grade ECG dataset
- **💾 Downloadable Results**: Export predictions as JSON or CSV

## 🏗️ Architecture

### HybridECGNet Model
- Residual Blocks with skip connections
- Channel & Spatial Attention (CBAM)
- Multi-scale feature extraction
- Batch Normalization & Dropout regularization
- Input: 224×224 RGB images
- Output: 3-class classification with confidence scores

## 📁 Project Structure

```
CardioVision-AI/
├── app.py                          # Main application
├── ECG_Digitization_Training.ipynb # Training notebook
├── flask_backend/                   # 🔥 Backend API (Updated)
│   ├── app.py                       # Flask server
│   ├── config.py                    # ✅ Updated configuration
│   ├── inference.py                 # ✅ Classification inference
│   ├── model_loader.py              # ✅ HybridECGNet loader
│   ├── preprocessing.py             # Image preprocessing
│   ├── ecg_model_final.pth          # ✅ Trained model (2.0.0)
│   └── best_model_advanced.pth      # Best checkpoint
├── frontend/                        # 🎨 Web Interface (Updated)
│   ├── index.html                   # ✅ Updated UI
│   ├── app.js                       # ✅ Classification display
│   └── styles.css                   # Styles
├── outputs/                         # Training outputs
├── checkpoints/                     # Model checkpoints
├── ecg_data/                        # Training data
├── start_backend.bat/.ps1           # ✅ Quick start scripts
├── start_frontend.bat               # ✅ Frontend launcher
├── INTEGRATION_GUIDE.md             # ✅ Detailed integration docs
└── README.md                        # This file

✅ = Recently updated for HybridECGNet integration
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch (CPU or GPU)
- Flask
- Modern web browser

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/SQUADRON-LEADER/CardioVision-AI.git
cd CardioVision-AI
```

2. **Setup virtual environment:**
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/Mac
```

3. **Install dependencies:**
```bash
pip install -r flask_backend/requirements.txt
```

### Running the Application

#### Option 1: Quick Start Scripts (Windows)

**Start Backend:**
```bash
# Using PowerShell
.\start_backend.ps1

# Or using CMD
start_backend.bat
```

**Start Frontend:**
```bash
start_frontend.bat
```

#### Option 2: Manual Start

**Backend Server:**
```bash
cd flask_backend
python app.py
```
→ Server runs at `http://localhost:5000`

**Frontend (Option A - Direct):**
- Simply open `frontend/index.html` in your browser

**Frontend (Option B - HTTP Server):**
```bash
cd frontend
python -m http.server 8000
```
→ Open `http://localhost:8000` in your browser

### Using the Application

1. **Open the frontend** in your web browser
2. **Upload an ECG image** (JPG/PNG formats supported)
3. **Click "Process ECG Image"**
4. **View results**:
   - Predicted condition class
   - Confidence percentage
   - Probability distribution for all classes
5. **Download results** as JSON or CSV

## 📊 Model Training

Train your own HybridECGNet model:

1. Open `ECG_Digitization_Training.ipynb` in Jupyter
2. Follow the cells sequentially
3. The notebook includes:
   - Data loading and preprocessing
   - HybridECGNet architecture definition
   - Training with attention mechanisms
   - Evaluation and visualization
   - Model export

## 🔌 API Reference

### POST `/api/v1/digitize`

Classify ECG image and return prediction.

**Request:**
```bash
curl -X POST http://localhost:5000/api/v1/digitize \
  -F "file=@ecg_image.png" \
  -F 'options={"remove_grid":true,"denoise":true}'
```

**Response (Classification):**
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
      "processing_time_seconds": 0.234
    },
    "quality_metrics": {
      "confidence_score": 85.5,
      "prediction_certainty": "high"
    }
  }
}
```

### GET `/health`
Health check endpoint.

### GET `/api/v1/info`
Get service information and model details.

## 🎯 Classification Classes

| Class | Description | Color Code |
|-------|-------------|------------|
| **Normal** | Regular heart rhythm, no abnormalities | 🟢 Green |
| **Abnormal Heartbeat** | Irregular heart rhythm patterns | 🟠 Orange |
| **Myocardial Infarction** | Heart attack indicators | 🔴 Red |

## 📈 Model Performance

- **Architecture**: HybridECGNet with CBAM Attention
- **Parameters**: ~5M trainable parameters
- **Input Size**: 224×224×3 RGB images
- **Training Dataset**: PTB-XL derived ECG images
- **Training Samples**: ~5,973 images
- **Test Accuracy**: ~70% (can be improved with more training)

## 🛠️ Technology Stack

- **Backend**: Flask, PyTorch, OpenCV, Pillow
- **Frontend**: HTML5, JavaScript, TailwindCSS, Canvas API
- **Model**: HybridECGNet (Custom CNN with Attention)
- **Deployment**: CPU/GPU compatible

## 📚 Documentation

- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** - Detailed integration documentation
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines
- **Jupyter Notebook** - Complete training pipeline with explanations

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**SQUADRON-LEADER**  
GitHub: [@SQUADRON-LEADER](https://github.com/SQUADRON-LEADER)

## 🙏 Acknowledgments

- PTB-XL dataset for ECG data
- PyTorch team for the deep learning framework
- Medical professionals for domain expertise

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) for detailed setup
- Review the training notebook for model details

---

**Status**: ✅ Production Ready  
**Last Updated**: February 2026  
**Version**: 2.0.0 (HybridECGNet)

- `POST /api/analyze` - Upload and analyze ECG image
- `GET /api/health` - Check service health status

## Technologies Used

- **PyTorch**: Deep learning framework
- **Flask**: Backend API framework
- **NumPy/Pandas**: Data processing
- **OpenCV**: Image processing
- **HTML/CSS/JavaScript**: Frontend interface

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Acknowledgments

- PTB-XL Dataset: Physionet
- Research papers on ECG classification
- Open-source deep learning community
