# CardioVision-AI

An advanced ECG analysis system using deep learning for automated heart condition detection and diagnosis.

## Overview

CardioVision-AI is a comprehensive electrocardiogram (ECG) analysis platform that leverages state-of-the-art deep learning models to detect and classify various cardiac conditions including:

- Normal ECG patterns
- Abnormal heartbeat patterns  
- Myocardial Infarction (Heart Attack)
- Post-MI history indicators

## Features

- **Automated ECG Analysis**: Upload ECG images and receive instant AI-powered analysis
- **Multi-condition Detection**: Trained on the PTB-XL dataset for robust classification
- **Web Interface**: User-friendly frontend for easy interaction
- **REST API**: Flask backend for integration with other systems
- **High Accuracy**: Advanced CNN architecture with attention mechanisms

## Project Structure

```
CardioVision-AI/
├── app.py                          # Main application
├── ECG_Digitization_Training.ipynb # Training notebook
├── convert_ptbxl_simple.py         # Data conversion utilities
├── convert_ptbxl_to_images.py      # Image generation pipeline
├── flask_backend/                   # Backend API
│   ├── app.py
│   ├── config.py
│   ├── inference.py
│   ├── model_loader.py
│   └── preprocessing.py
├── frontend/                        # Web interface
│   ├── index.html
│   ├── app.js
│   └── styles.css
└── checkpoints/                     # Model checkpoints
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SQUADRON-LEADER/CardioVision-AI.git
cd CardioVision-AI
```

2. Create a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r flask_backend/requirements.txt
```

## Usage

### Running the Flask Backend

```bash
cd flask_backend
python app.py
```

The API will be available at `http://localhost:5000`

### Running the Frontend

Open `frontend/index.html` in a web browser or serve it using a local server.

## Model Training

The model is trained using the PTB-XL dataset. To train your own model:

1. Download the PTB-XL dataset
2. Run the data conversion scripts
3. Execute the training notebook: `ECG_Digitization_Training.ipynb`

## API Endpoints

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
