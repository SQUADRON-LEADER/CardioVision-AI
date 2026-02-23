# CardioVision AI - Frontend

Modern, responsive frontend for ECG digitization service.

## Features

- 🎨 Clean, professional design with dark theme
- 🌊 WebGL fluid animation background (Unicorn Studio)
- 📱 Fully responsive (mobile, tablet, desktop)
- 🖼️ Drag & drop file upload
- 📊 Real-time signal visualization
- 💾 Export to JSON/CSV formats
- ⚡ Fast and lightweight (no heavy frameworks)
- 🎭 Smooth animations and transitions
- 🎯 Iconify icons for consistent design

## Tech Stack

- HTML5
- CSS3 (Tailwind CSS via CDN)
- Vanilla JavaScript
- Canvas API for signal visualization

## Setup

### 1. Start Flask Backend

```bash
cd flask_backend
python app.py
```

Backend will run on `http://localhost:5000`

### 2. Serve Frontend

**Option A: Python HTTP Server**
```bash
cd frontend
python -m http.server 8000
```

**Option B: Node.js HTTP Server**
```bash
cd frontend
npx http-server -p 8000
```

**Option C: VS Code Live Server**
- Install "Live Server" extension
- Right-click `index.html` → "Open with Live Server"

### 3. Open Browser

Navigate to `http://localhost:8000`

## Usage

1. Click or drag & drop an ECG image
2. Click "Process ECG Image"
3. View digitized signals and metrics
4. Download results as JSON or CSV

## API Configuration

Edit `app.js` to change backend URL:

```javascript
const API_URL = 'http://localhost:5000/api/v1';
```

## File Structure

```
frontend/
├── index.html      # Main HTML file
├── styles.css      # Custom styles
├── app.js          # JavaScript logic
└── README.md       # This file
```

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

## Customization

### Colors

Edit Tailwind classes in `index.html`:
- Primary: `red-600` → change to your color
- Background: `#050505` → change in `styles.css`

### Logo

Replace the heart SVG icon in the navigation with your logo.

## Production Deployment

### Build for Production

1. Minify CSS/JS
2. Optimize images
3. Enable CORS on backend
4. Use HTTPS

### Deploy Options

- **Vercel**: `vercel deploy`
- **Netlify**: Drag & drop `frontend/` folder
- **GitHub Pages**: Push to `gh-pages` branch
- **AWS S3**: Upload as static website

## Troubleshooting

### CORS Errors

Add to Flask backend (`app.py`):
```python
from flask_cors import CORS
CORS(app, origins=['http://localhost:8000'])
```

### Backend Not Responding

1. Check Flask is running: `http://localhost:5000/health`
2. Verify API_URL in `app.js`
3. Check browser console for errors

### File Upload Fails

1. Check file size < 16MB
2. Verify file is image format (JPG, PNG)
3. Check backend logs for errors

## License

MIT License
