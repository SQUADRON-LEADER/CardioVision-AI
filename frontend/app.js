// API Configuration
const API_URL = 'http://localhost:5000/api/v1';

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const previewImage = document.getElementById('previewImage');
const processBtn = document.getElementById('processBtn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const processingTime = document.getElementById('processingTime');
const qualityScore = document.getElementById('qualityScore');
const leadsCount = document.getElementById('leadsCount');
const signalChart = document.getElementById('signalChart');
const downloadJSON = document.getElementById('downloadJSON');
const downloadCSV = document.getElementById('downloadCSV');
const newUpload = document.getElementById('newUpload');

let currentFile = null;
let currentResults = null;

// Drop zone click handler
dropZone.addEventListener('click', () => {
    fileInput.click();
});

// File input change handler
fileInput.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
});

// Drag and drop handlers
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    handleFile(e.dataTransfer.files[0]);
});

// Handle file selection
function handleFile(file) {
    if (!file) return;
    
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file');
        return;
    }
    
    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        alert('File size must be less than 16MB');
        return;
    }
    
    currentFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        preview.classList.remove('hidden');
        results.classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

// Process ECG image
processBtn.addEventListener('click', async () => {
    if (!currentFile) return;
    
    // Show loading
    loading.classList.remove('hidden');
    preview.classList.add('hidden');
    results.classList.add('hidden');
    
    try {
        // Create form data
        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('options', JSON.stringify({
            remove_grid: true,
            denoise: true,
            output_format: 'json'
        }));
        
        // Send request to backend
        const response = await fetch(`${API_URL}/digitize`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            currentResults = data.data;
            displayResults(data.data);
        } else {
            throw new Error(data.error || 'Processing failed');
        }
        
    } catch (error) {
        console.error('Error:', error);
        alert(`Error processing ECG: ${error.message}`);
        preview.classList.remove('hidden');
    } finally {
        loading.classList.add('hidden');
    }
});

// Display results
function displayResults(data) {
    // Update metrics
    processingTime.textContent = `${data.metadata.processing_time_seconds.toFixed(2)}s`;
    
    // Check if this is classification or digitization
    if (data.metadata.task === 'classification' && data.prediction) {
        // Classification results
        displayClassificationResults(data);
    } else if (data.signals) {
        // Digitization results (legacy)
        displayDigitizationResults(data);
    } else {
        console.error('Unknown result format:', data);
        alert('Received unexpected result format from server');
    }
    
    // Show results
    results.classList.remove('hidden');
}

// Display classification results
function displayClassificationResults(data) {
    const prediction = data.prediction;
    
    // Update quality score with confidence
    qualityScore.textContent = prediction.confidence + '%';
    
    // Update leads count with class name
    leadsCount.textContent = prediction.class;
    
    // Draw classification visualization
    drawClassificationChart(prediction);
}

// Display digitization results
function displayDigitizationResults(data) {
    qualityScore.textContent = data.quality_metrics?.quality_score 
        ? (data.quality_metrics.quality_score * 100).toFixed(1) + '%'
        : 'N/A';
    leadsCount.textContent = data.metadata.num_leads;
    
    // Draw signal chart
    drawSignalChart(data.signals);
}

// Draw classification chart
function drawClassificationChart(prediction) {
    const ctx = signalChart.getContext('2d');
    const width = signalChart.width;
    const height = signalChart.height;
    
    // Clear canvas
    ctx.fillStyle = '#050505';
    ctx.fillRect(0, 0, width, height);
    
    // Title
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 24px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('ECG Classification Result', width / 2, 50);
    
    // Predicted class
    ctx.font = 'bold 32px Inter';
    const classColors = {
        'Normal': '#10b981',
        'Abnormal Heartbeat': '#f59e0b',
        'Myocardial Infarction': '#ef4444'
    };
    ctx.fillStyle = classColors[prediction.class] || '#ffffff';
    ctx.fillText(prediction.class, width / 2, 120);
    
    // Confidence
    ctx.font = '20px Inter';
    ctx.fillStyle = '#9ca3af';
    ctx.fillText(`Confidence: ${prediction.confidence}%`, width / 2, 160);
    
    // Probability distribution bars
    ctx.textAlign = 'left';
    ctx.font = '16px Inter';
    const startY = 220;
    const barMaxWidth = width - 100;
    const barHeight = 30;
    const barSpacing = 50;
    
    Object.entries(prediction.probability_distribution).forEach(([className, prob], idx) => {
        const y = startY + (idx * barSpacing);
        
        // Label
        ctx.fillStyle = '#ffffff';
        ctx.fillText(className, 50, y);
        
        // Bar background
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(50, y + 5, barMaxWidth, barHeight);
        
        // Bar foreground
        const barWidth = prob * barMaxWidth;
        ctx.fillStyle = classColors[className] || '#6366f1';
        ctx.fillRect(50, y + 5, barWidth, barHeight);
        
        // Percentage text
        ctx.fillStyle = '#ffffff';
        ctx.fillText(`${(prob * 100).toFixed(1)}%`, 50 + barWidth + 10, y + 25);
    });
}

// Draw signal chart (simple visualization)
function drawSignalChart(signals) {
    const ctx = signalChart.getContext('2d');
    const width = signalChart.width;
    const height = signalChart.height;
    
    // Clear canvas
    ctx.fillStyle = '#050505';
    ctx.fillRect(0, 0, width, height);
    
    // Draw grid
    ctx.strokeStyle = '#1a1a1a';
    ctx.lineWidth = 1;
    for (let i = 0; i < width; i += 20) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i, height);
        ctx.stroke();
    }
    for (let i = 0; i < height; i += 20) {
        ctx.beginPath();
        ctx.moveTo(0, i);
        ctx.lineTo(width, i);
        ctx.stroke();
    }
    
    // Draw signals (show first 3 leads as example)
    const leadNames = ['I', 'II', 'III'];
    const colors = ['#ef4444', '#f97316', '#eab308'];
    const spacing = height / 4;
    
    leadNames.forEach((lead, idx) => {
        if (!signals[lead]) return;
        
        const signal = signals[lead];
        const yOffset = spacing * (idx + 1);
        const xScale = width / signal.length;
        const yScale = spacing / 4;
        
        ctx.strokeStyle = colors[idx];
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        signal.forEach((value, i) => {
            const x = i * xScale;
            const y = yOffset - (value * yScale);
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
        
        // Draw lead label
        ctx.fillStyle = colors[idx];
        ctx.font = '12px Inter';
        ctx.fillText(`Lead ${lead}`, 10, yOffset - spacing / 2 + 5);
    });
}

// Download JSON
downloadJSON.addEventListener('click', () => {
    if (!currentResults) return;
    
    const dataStr = JSON.stringify(currentResults, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `ecg_signals_${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
});

// Download CSV
downloadCSV.addEventListener('click', () => {
    if (!currentResults) return;
    
    // Check if this is classification or digitization
    if (currentResults.prediction) {
        // Classification CSV
        const prediction = currentResults.prediction;
        let csv = 'ECG Classification Results\n\n';
        csv += 'Predicted Class,' + prediction.class + '\n';
        csv += 'Confidence (%),' + prediction.confidence + '\n\n';
        csv += 'Class,Probability\n';
        
        Object.entries(prediction.probability_distribution).forEach(([className, prob]) => {
            csv += className + ',' + (prob * 100).toFixed(2) + '%\n';
        });
        
        csv += '\nMetadata\n';
        csv += 'Processing Time (s),' + currentResults.metadata.processing_time_seconds + '\n';
        csv += 'Timestamp,' + currentResults.metadata.timestamp + '\n';
        
        const dataBlob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `ecg_classification_${Date.now()}.csv`;
        link.click();
        URL.revokeObjectURL(url);
    } else if (currentResults.signals) {
        // Digitization CSV (original)
        const signals = currentResults.signals;
        const leadNames = Object.keys(signals);
        const signalLength = signals[leadNames[0]].length;
        
        let csv = 'Time(s),' + leadNames.join(',') + '\n';
        
        for (let i = 0; i < signalLength; i++) {
            const time = (i / currentResults.metadata.sampling_rate_hz).toFixed(4);
            const values = leadNames.map(lead => signals[lead][i].toFixed(6));
            csv += time + ',' + values.join(',') + '\n';
        }
        
        const dataBlob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `ecg_signals_${Date.now()}.csv`;
        link.click();
        URL.revokeObjectURL(url);
    }
});

// New upload
newUpload.addEventListener('click', () => {
    currentFile = null;
    currentResults = null;
    fileInput.value = '';
    preview.classList.add('hidden');
    results.classList.add('hidden');
});

// Smooth scroll for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });
});
