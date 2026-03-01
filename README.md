# CardioVision-AI

An advanced ECG analysis platform with **dual-mode AI** - Classification for diagnosis + Digitization for signal extraction.

## 🎯 Overview

CardioVision-AI is a comprehensive electrocardiogram (ECG) analysis platform offering two powerful AI modes:

### 🩺 Classification Mode
Diagnose cardiac conditions using HybridECGNet architecture:
- ✅ **Normal ECG patterns**
- ✅ **Abnormal heartbeat patterns**
- ✅ **Myocardial Infarction (Heart Attack)**

### 📊 Digitization Mode  
Extract 12-lead time-series signals from ECG images:
- ✅ **12-Lead Signal Extraction** (I, II, III, aVR, aVL, aVF, V1-V6)
- ✅ **PhysioNet Competition Ready** (PTB-XL dataset compatible)
- ✅ **Quality Metrics** (SNR, signal validation)
- ✅ **Multiple Export Formats** (JSON, CSV, WFDB)

## ✨ Features

- **🤖 Dual AI Models**: Classification (HybridECGNet) + Digitization (AdvancedECGNet)
- **📊 Real-time Processing**: Upload ECG images and receive instant results
- **🎨 Modern Web Interface**: Interactive signal visualization with Plotly.js
- **🔌 REST API**: Flask backend with unified model management
- **📈 High Performance**: GPU acceleration support, <2s inference
- **💾 Export Options**: JSON, CSV downloads for further analysis
- **📓 Jupyter Training**: Complete training notebook for digitization model

## 🏗️ Architecture

### Classification Mode: HybridECGNet
- **Architecture**: ResNet34 + CBAM Attention
- **Parameters**: ~35M parameters
- **Input**: 224×224 RGB images
- **Output**: 3-class classification with confidence scores
- **Features**: Residual blocks, channel/spatial attention, dropout regularization

### Digitization Mode: AdvancedECGDigitizationModel
- **Architecture**: ResNet50 encoder + Multi-head Attention + Per-lead Decoders
- **Parameters**: ~55M parameters  
- **Input**: 224×224 RGB images
- **Output**: 12 leads × 1000 samples per lead
- **Loss Function**: Combined MSE +         # Legacy main application
├── ECG_Digitization_Complete_Training.ipynb # 🆕 Full training notebook
├── ECG_Digitization_Model.py               # 🆕 Digitization model architectures
├── ECG_Digitization_Dataset.py             # 🆕 PTB-XL dataset loaders
├── train_digitization.py                   # 🆕 Training script
├── backend/                          # 🔥 Backend API (Unified)
│   ├── app.py                              # ✅ Flask server (dual mode)
│   ├── config.py                           # ✅ MODEL_PATHS configuration
│   ├── model_loader_unified.py             # 🆕 Unified model manager
│   ├── inference_unified.py                # 🆕 Unified inference engine
│   ├── inference.py                        # Legacy classification inference
│   ├── model_loader.py                     # Legacy HybridECGNet loader
│   ├── preprocessing.py                    # Image preprocessing
│   ├── ecg_model_final.pth                 # Classification model
│   └── best_model_digitization.pth         # 🆕 Digitization model (train first)
├── frontend/                               # 🎨 Web Interface (Dual Mode)
│   ├── index_enhanced.html                 # 🆕 Dual-mode UI with task selector
│   ├── app_enhanced.js                     # 🆕 Dual-mode logic + Plotly signals
│   ├── index.html                          # Legacy UI (classification only)
│   ├── app.js                              # Legacy JavaScript
│   └── styles.css                          # Shared styles
├── outputs/                                # Training outputs
├── checkpoints/                            # Model checkpoints
├── ecg_data/                               # Classification training data
├── ptb-xl-.../                             # 🆕 PTB-XL dataset for digitization
├── start_full_system.bat                   # 🆕 Launch backend + frontend
├── start_digitization_training.bat         # 🆕 Quick training start
├── TESTING_GUIDE.md                        # 🆕 Comprehensive testing guide
├── ECG_DIGITIZATION_GUIDE.md               # 🆕 Complete digitization docs
├── PROJECT_SUMMARY.md                      # 🆕 Quick reference
└── README.md                               # This file

🆕 = New for dual-mode system (v3.0.0)
✅ = Updated for dual-mode supportkpoints
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
- **For Digitization**: PTB-XL dataset, wfdb package

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
pip install -r backend/requirements.txt
pip install wfdb  # For digitization mode
```

### Running the Full System (Both Modes)

#### Option 1: One-Click Launcher (Windows)
```bash
start_full_system.bat
```
This will:
- Start Flask backend at `http://localhost:5000`
- Open enhanced frontend with task selector
- Display testing instructions

#### Option 2: Manual Start

**Backend Server:**
```bash
cd backend
python app.py
```
→ Server runs at `http://localhost:5000`

**Frontend:**
- **Enhanced UI** (Dual Mode): Open `frontend/index_enhanced.html`
- **Legacy UI** (Classification Only): Open `frontend/index.html`

### Using the Application

#### Classification Mode
1. **Select "Classification"** in the task selector
2. **Upload an ECG image** from `ecg_data/` folders
3. **Click "Process ECG"**
4. **View results**:
   - Diagnosis: Normal / Abnormal / MI
   - Confidence percentage
   - Probability distribution
5. **Download** as JSON

#### Digitization Mode
1. **Select "Digitization"** in the task selector
2. **Upload any ECG image** (printout or PTB-XL)
3. **Click "Process ECG"**
4. **View results**:
   - Interactive 12-lead signal plot
   - Quality metrics (SNR, sampling rate)
   - Signal statistics
5. **Export** as JSON or CSV

## 📊 Training Models

### Classification Model (HybridECGNet)
Already trained! Model file: `backend/ecg_model_final.pth`

### Digitization Model (Train Required)

#### Option 1: Jupyter Notebook (Recommended)
```bash
jupyter notebook ECG_Digitization_Complete_Training.ipynb
```
Run all cells to train and save model.

#### Option 2: Python Script
```bash
python train_digitization.py
```

#### Quick Training Start
```bash
start_digitization_training.bat  # Windows
```

**After Training:**
```bash
# Copy trained model to backend
copy checkpoints\best_model_digitization.pth backend\
```

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for detailed training instructions.

## 🔌 API Reference

### POST `/api/v1/digitize`

Process ECG image for classification or digitization.

#### Classification Request
```bash
curl -X POST http://localhost:5000/api/v1/digitize \
  -F "file=@ecg_image.png" \
  -F "task=classification" \
  -F 'options={"remove_grid":true,"denoise":true}'
```

**Response (Classification):**
```json
{
  "success": true,
  "data": {
    "request_id": "uuid",
    "task": "classification",
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
      "model_version": "3.0.0",
      "processing_time_seconds": 0.234
    },
    "quality_metrics": {
      "confidence_score": 85.5,
      "prediction_certainty": "high"
    }
  }
}
```

#### Digitization Request
```bash
curl -X POST http://localhost:5000/api/v1/digitize \
  -F "file=@ecg_image.png" \
  -F "task=digitization" \
  -F 'options={"remove_grid":true,"denoise":true,"output_format":"json"}'
```

**Response (Digitization):**
```json
{
  "success": true,
  "data": {
    "request_id": "uuid",
    "task": "digitization",
    "signals": {
      "I": [0.12, 0.15, 0.18, ...],  // 1000 samples
      "II": [0.23, 0.21, 0.24, ...],
      "III": [-0.11, -0.06, ...],
      "aVR": [-0.17, -0.18, ...],
      "aVL": [0.12, 0.10, ...],
      "aVF": [0.06, 0.08, ...],
      "V1": [0.04, 0.03, ...],
      "V2": [0.15, 0.18, ...],
      "V3": [0.32, 0.35, ...],
      "V4": [0.45, 0.48, ...],
      "V5": [0.38, 0.40, ...],
      "V6": [0.28, 0.30, ...]
    },
    "quality_metrics": {
      "average_snr_db": 18.5,
      "overall_quality": "Good",
      "per_lead_quality": {...}
    },
    "metadata": {
      "task": "digitization",
      "model_type": "AdvancedECGDigitizationModel",
      "model_version": "3.0.0",
      "processing_time_seconds": 1.234,
      "num_leads": 12,
      "signal_length": 1000,
      "sampling_rate_hz": 100
    }
  }
}
```

### GET `/health`
Health check endpoint.

### GET `/api/v1/info`
Get service information, available tasks, and model details.

**Response:**
```json
{
  "service": "CardioVision AI - ECG Analysis",
  "version": "3.0.0",
  "available_tasks": ["classification", "digitization"],
  "models": {
    "classification": {
      "type": "HybridECGNet",
      "classes": ["Normal", "Abnormal Heartbeat", "Myocardial Infarction"]
    },
    "digitization": {
      "type": "AdvancedECGDigitizationModel",
      "leads": 12,
      "output_length": 1000
    }
  }
}
```

## 🎯 Classification Classes

| Class | Description | Color Code |
|-------|-------------|------------|
| **Normal** | Regular heart rhythm, no abnormalities | 🟢 Green |
| **Abnormal Heartbeat** | Irregular heart rhythm patterns | 🟠 Orange |
| **Myocardial Infarction** | Heart attack indicators | 🔴 Red |

## 📈 Model Performance

### Classification Model
- **Architecture**: HybridECGNet with CBAM Attention
- **Parameters**: ~35M trainable parameters
- **Input Size**: 224×224×3 RGB images
- **Training Dataset**: Custom ECG image dataset
- **Test Accuracy**: ~70-85% (varies by class)

### Digitization Model
- **Architecture**: ResNet50 + Multi-head Attention + Per-lead Decoders
- **Parameters**: ~55M trainable parameters
- **Input Size**: 224×224×3 RGB images
- **Training Dataset**: PTB-XL (21,837 ECG recordings)
- **Output**: 12 leads × 1000 samples (100 Hz sampling)
- **Quality**: Average SNR >15 dB on validation set

## 🛠️ Technology Stack

- **Backend**: Flask, PyTorch, OpenCV, Pillow, wfdb
- **Frontend**: HTML5, JavaScript, TailwindCSS, Plotly.js
- **Models**: 
  - HybridECGNet (Classification)
  - AdvancedECGDigitizationModel (Signal Extraction)
- **Deployment**: CPU/GPU compatible
- **Training**: Jupyter Notebooks with full pipeline

## 📚 Documentation

- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Complete testing instructions for both modes
- **[ECG_DIGITIZATION_GUIDE.md](ECG_DIGITIZATION_GUIDE.md)** - Digitization system documentation
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Quick reference guide
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines
- **Jupyter Notebooks** - Training pipelines with explanations

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**SQUADRON-LEADER**  
GitHub: [@SQUADRON-LEADER](https://github.com/SQUADRON-LEADER)

## 🙏 Acknowledgments

- PTB-XL dataset (PhysioNet) for ECG data
- PyTorch team for the deep learning framework
- Medical professionals for domain expertise
- ECG-Digitiser project for frontend inspiration

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check [TESTING_GUIDE.md](TESTING_GUIDE.md) for setup instructions
- Review [ECG_DIGITIZATION_GUIDE.md](ECG_DIGITIZATION_GUIDE.md) for detailed docs
- Explore the training notebooks for model details

---

**Status**: ✅ Production Ready (Classification) | 🔄 Training Required (Digitization)  
**Last Updated**: January 2025
**Version**: 3.0.0 (Dual-Mode System)
