# Deployment Guide

## Production Deployment

### Prerequisites
- Python 3.8+
- Virtual environment
- Optional: GPU (CUDA 11.0+)

### Setup
1. Clone repository
2. Create virtual environment
3. Install requirements: pip install -r backend/requirements.txt
4. Set environment variables in .env
5. Run: python backend/app.py

### Environment Variables
- HOST=0.0.0.0
- PORT=5000
- DEBUG=False
- DEVICE=cpu (or cuda)
- DEFAULT_TASK=pipeline

### Docker
```
docker build -t cardiovision-ai .
docker run -p 5000:5000 cardiovision-ai
```

### Performance
- Use GPU for batch processing
- Monitor memory for large batches
- Check health endpoint: GET /health
