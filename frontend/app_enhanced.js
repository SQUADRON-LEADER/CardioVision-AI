// API Configuration
const API_URL = 'http://localhost:5000/api/v1';

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const previewImage = document.getElementById('previewImage');
const taskSelector = document.getElementById('taskSelector');
const processBtn = document.getElementById('processBtn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const downloadJSON = document.getElementById('downloadJSON');
const downloadCSV = document.getElementById('downloadCSV');
const newUpload = document.getElementById('newUpload');

//Results containers
const classificationResults = document.getElementById('classificationResults');
const digitizationResults = document.getElementById('digitizationResults');

let currentFile = null;
let currentResults = null;
let currentTask = 'digitization';

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Set initial task
    if (taskSelector) {
        taskSelector.addEventListener('change', (e) => {
            currentTask = e.target.value;
            console.log('Task changed to:', currentTask);
        });
        currentTask = taskSelector.value;
    }
});

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
        formData.append('task', currentTask);
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
            throw new Error(`Server error: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.error || 'Processing failed');
        }
        
        currentResults = result.data;
        
        // Hide loading
        loading.classList.add('hidden');
        
        // Display results based on task
        displayResults(currentResults);
        
    } catch (error) {
        console.error('Error:', error);
        loading.classList.add('hidden');
        alert(`Error: ${error.message}`);
    }
});

// Display results
function displayResults(data) {
    results.classList.remove('hidden');
    
    // Update metadata
    if (data.metadata) {
        document.getElementById('processingTime').textContent = 
            `${data.metadata.processing_time_seconds.toFixed(2)}s`;
    }
    
    // Display based on task
    if (data.task === 'classification') {
        displayClassificationResults(data);
    } else if (data.task === 'digitization') {
        displayDigitizationResults(data);
    }
}

// Display classification results
function displayClassificationResults(data) {
    if (classificationResults) {
        classificationResults.classList.remove('hidden');
    }
    if (digitizationResults) {
        digitizationResults.classList.add('hidden');
    }
    
    const pred = data.prediction;
    
    // Update prediction
    const predictionClass = document.getElementById('predictionClass');
    const predictionConfidence = document.getElementById('predictionConfidence');
    
    if (predictionClass) {
        predictionClass.textContent = pred.class;
        
        // Color code based on class
        if (pred.class === 'Normal') {
            predictionClass.className = 'text-2xl font-bold text-green-400';
        } else if (pred.class === 'Abnormal Heartbeat') {
            predictionClass.className = 'text-2xl font-bold text-yellow-400';
        } else {
            predictionClass.className = 'text-2xl font-bold text-red-400';
        }
    }
    
    if (predictionConfidence) {
        predictionConfidence.textContent = `${pred.confidence}%`;
    }
    
    // Update probability distribution
    const probBarsContainer = document.getElementById('probabilityBars');
    if (probBarsContainer && pred.probability_distribution) {
        probBarsContainer.innerHTML = '';
        
        Object.entries(pred.probability_distribution).forEach(([className, prob]) => {
            const percentage = (prob * 100).toFixed(1);
            const barHTML = `
                <div class="mb-3">
                    <div class="flex justify-between mb-1">
                        <span class="text-sm text-neutral-400">${className}</span>
                        <span class="text-sm text-neutral-300 font-semibold">${percentage}%</span>
                    </div>
                    <div class="w-full bg-neutral-800 rounded-full h-2">
                        <div class="bg-gradient-to-r from-red-600 to-red-500 h-2 rounded-full transition-all duration-500" 
                             style="width: ${percentage}%"></div>
                    </div>
                </div>
            `;
            probBarsContainer.innerHTML += barHTML;
        });
    }
    
    // Update quality metrics
    if (data.quality_metrics) {
        const qualityScore = document.getElementById('qualityScore');
        if (qualityScore) {
            qualityScore.textContent = data.quality_metrics.confidence_score + '%';
        }
    }
}

// Display digitization results
function displayDigitizationResults(data) {
    if (classificationResults) {
        classificationResults.classList.add('hidden');
    }
    if (digitizationResults) {
        digitizationResults.classList.remove('hidden');
    }
    
    // Update metadata
    if (data.metadata) {
        const samplingRate = document.getElementById('samplingRate');
        const signalLength = document.getElementById('signalLength');
        
        if (samplingRate) {
            samplingRate.textContent = `${data.metadata.sampling_rate_hz} Hz`;
        }
        if (signalLength) {
            signalLength.textContent = `${data.metadata.signal_length} samples`;
        }
    }
    
    // Update quality
    if (data.quality_metrics) {
        const qualityScore = document.getElementById('qualityScore');
        const avgSNR = document.getElementById('avgSNR');
        
        if (qualityScore) {
            qualityScore.textContent = data.quality_metrics.overall_quality || 'Good';
        }
        if (avgSNR && data.quality_metrics.average_snr_db) {
            avgSNR.textContent = `${data.quality_metrics.average_snr_db.toFixed(1)} dB`;
        }
    }
    
    // Plot signals
    if (data.signals) {
        plotECGSignals(data.signals);
    }
}

// Plot ECG signals using Plotly
function plotECGSignals(signals) {
    const leadNames = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];
    const colors = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6', '#ec4899',
                    '#06b6d4', '#84cc16', '#f97316', '#14b8a6', '#a855f7', '#eab308'];
    
    // Create container for plots
    const signalChart = document.getElementById('signalChart');
    if (!signalChart) return;
    
    signalChart.innerHTML = '<div id="ecgPlot" class="w-full" style="height: 800px;"></div>';
    
    // Prepare data for Plotly
    const traces = [];
    
    leadNames.forEach((lead, idx) => {
        if (signals[lead]) {
            const signal = signals[lead];
            const x = Array.from({length: signal.length}, (_, i) => i);
            
            traces.push({
                x: x,
                y: signal,
                type: 'scatter',
                mode: 'lines',
                name: `Lead ${lead}`,
                line: {
                    color: colors[idx],
                    width: 1.5
                },
                yaxis: `y${idx + 1}`
            });
        }
    });
    
    // Layout with stacked subplots
    const layout = {
        title: {
            text: '12-Lead ECG Signals',
            font: {
                color: '#fff',
                size: 20,
                family: 'Inter, sans-serif'
            }
        },
        paper_bgcolor: '#0a0a0a',
        plot_bgcolor: '#050505',
        showlegend: true,
        legend: {
            orientation: 'h',
            y: -0.15,
            font: { color: '#999' }
        },
        grid: {
            rows: 12,
            columns: 1,
            pattern: 'independent',
            roworder: 'top to bottom'
        },
        xaxis: {
            title: { text: 'Sample', font: { color: '#999' } },
            gridcolor: '#1a1a1a',
            color: '#666'
        },
        margin: { t: 60, b: 100, l: 60, r: 40 }
    };
    
    // Configure each y-axis
    for (let i = 0; i < 12; i++) {
        layout[`yaxis${i === 0 ? '' : i + 1}`] = {
            title: { text: leadNames[i], font: { color: colors[i], size: 10 } },
            gridcolor: '#1a1a1a',
            color: '#666',
            zeroline: true,
            zerolinecolor: '#333'
        };
    }
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d']
    };
    
    Plotly.newPlot('ecgPlot', traces, layout, config);
}

// Download JSON
if (downloadJSON) {
    downloadJSON.addEventListener('click', () => {
        if (!currentResults) return;
        
        const dataStr = JSON.stringify(currentResults, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `ecg_${currentResults.request_id}.json`;
        link.click();
        URL.revokeObjectURL(url);
    });
}

// Download CSV
if (downloadCSV) {
    downloadCSV.addEventListener('click', async () => {
        if (!currentResults || !currentResults.signals) {
            alert('No signals data available for CSV export');
            return;
        }
        
        // Convert signals to CSV
        const signals = currentResults.signals;
        const leadNames = Object.keys(signals);
        const numSamples = signals[leadNames[0]].length;
        
        // Header
        let csv = leadNames.join(',') + '\n';
        
        // Data rows
        for (let i = 0; i < numSamples; i++) {
            const row = leadNames.map(lead => signals[lead][i]).join(',');
            csv += row + '\n';
        }
        
        // Download
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `ecg_${currentResults.request_id}.csv`;
        link.click();
        URL.revokeObjectURL(url);
    });
}

// New upload
if (newUpload) {
    newUpload.addEventListener('click', () => {
        currentFile = null;
        currentResults = null;
        preview.classList.add('hidden');
        results.classList.add('hidden');
        fileInput.value = '';
    });
}
