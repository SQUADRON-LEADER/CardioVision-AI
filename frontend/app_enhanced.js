// API Configuration
const isBackendOrigin = window.location.port === '5000';
const API_URL = isBackendOrigin
    ? `${window.location.origin}/api/v1`
    : 'http://localhost:5000/api/v1';

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const previewImage = document.getElementById('previewImage');
const taskInputs = document.querySelectorAll('input[name="task"]');
const processBtn = document.getElementById('processBtn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const errorAlert = document.getElementById('errorAlert');
const errorMessage = document.getElementById('errorMessage');
const closeError = document.getElementById('closeError');
const downloadJSON = document.getElementById('downloadJSON');
const downloadCSV = document.getElementById('downloadCSV');
const newUpload = document.getElementById('newUpload');

//Results containers
const classificationResults = document.getElementById('classificationResults');
const digitizationResults = document.getElementById('digitizationResults');

let currentFile = null;
let currentResults = null;
let currentTask = 'pipeline';
let isProcessing = false;
let activeFileToken = 0;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    if (taskInputs && taskInputs.length > 0) {
        taskInputs.forEach((input) => {
            input.addEventListener('change', (e) => {
                currentTask = e.target.value;
            });
        });
        const checked = document.querySelector('input[name="task"]:checked');
        currentTask = checked ? checked.value : 'pipeline';
    }
    
    // Close error button
    if (closeError) {
        closeError.addEventListener('click', (e) => {
            e.preventDefault();
            errorAlert.classList.add('hidden');
        });
    }
});

// Helper function to show error
function showError(message) {
    if (errorMessage) {
        errorMessage.textContent = message;
    }
    if (errorAlert) {
        errorAlert.classList.remove('hidden');
    }
}

// Helper function to hide error
function hideError() {
    if (errorAlert) {
        errorAlert.classList.add('hidden');
    }
}

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
        showError('Please upload an image file');
        return;
    }
    
    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        showError('File size must be less than 16MB');
        return;
    }
    
    currentFile = file;
    currentResults = null;
    isProcessing = false;
    if (processBtn) {
        processBtn.disabled = false;
    }
    activeFileToken += 1;
    const fileToken = activeFileToken;

    // Hide stale results and errors
    results.classList.add('hidden');
    loading.classList.add('hidden');
    hideError();
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        // Ignore stale reads from older selections.
        if (fileToken !== activeFileToken) return;

        previewImage.src = e.target.result;

        // Don't let late image-read callbacks overwrite post-processing UI.
        if (!isProcessing) {
            preview.classList.remove('hidden');
        }
    };
    reader.readAsDataURL(file);
}

// Process ECG image
processBtn.addEventListener('click', async (e) => {
    e.preventDefault();
    if (!currentFile) return;
    if (isProcessing) return;
    isProcessing = true;
    if (processBtn) {
        processBtn.disabled = true;
    }
    
    // Show loading and hide other elements
    loading.classList.remove('hidden');
    preview.classList.add('hidden');
    results.classList.add('hidden');
    hideError();
    
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

        const responseBody = await response.json().catch(() => null);
        
        if (!response.ok) {
            const backendMessage = responseBody?.error || responseBody?.message;
            throw new Error(backendMessage || `Server error: ${response.status}`);
        }

        const result = responseBody;
        
        if (!result.success) {
            throw new Error(result.error || 'Processing failed');
        }
        
        currentResults = result.data;
        
        // Hide loading
        loading.classList.add('hidden');
        isProcessing = false;
        if (processBtn) {
            processBtn.disabled = false;
        }
        
        // Display results based on task
        displayResults(currentResults);
        
    } catch (error) {
        console.error('Error:', error);
        loading.classList.add('hidden');
        isProcessing = false;
        if (processBtn) {
            processBtn.disabled = false;
        }
        // Restore preview so UI doesn't appear empty after a failed render/request.
        preview.classList.remove('hidden');
        results.classList.add('hidden');
        showError(error.message);
    }
});

// Display results
function displayResults(data) {
    if (!data || typeof data !== 'object') {
        throw new Error('Invalid response payload from server');
    }

    results.classList.remove('hidden');
    preview.classList.add('hidden');
    
    // Update metadata
    if (data.metadata) {
        const processingTimeEl = document.getElementById('processingTime');
        const processingSeconds = Number(
            data.metadata.processing_time_seconds ?? data.metadata.processing_time
        );
        if (processingTimeEl) {
            processingTimeEl.textContent = Number.isFinite(processingSeconds)
                ? `${processingSeconds.toFixed(2)}s`
                : '--';
        }
    }
    
    const modeDisplay = document.getElementById('modeDisplay');
    const inferredTask = data.task
        || data.metadata?.task
        || ((data.prediction && data.signals) ? 'pipeline' : (data.prediction ? 'classification' : (data.signals ? 'digitization' : 'pipeline')));
    if (modeDisplay) {
        modeDisplay.textContent = inferredTask;
    }

    // Display based on task
    if ((inferredTask === 'pipeline') || (data.prediction && data.signals)) {
        displayPipelineResults(data);
    } else if (inferredTask === 'classification') {
        displayClassificationResults(data);
    } else if (inferredTask === 'digitization') {
        displayDigitizationResults(data);
    }
}

function displayPipelineResults(data) {
    // Show both sections together.
    if (classificationResults) {
        classificationResults.classList.remove('hidden');
    }
    if (digitizationResults) {
        digitizationResults.classList.remove('hidden');
    }

    displayClassificationResults(data, { keepDigitizationVisible: true });
    displayDigitizationResults(data, { keepClassificationVisible: true });

    const qualityScore = document.getElementById('qualityScore');
    const clsConf = data.prediction?.confidence;
    const sigQuality = data.quality_metrics?.overall_quality;
    if (qualityScore) {
        if (clsConf && sigQuality) {
            qualityScore.textContent = `${clsConf}% / ${sigQuality}`;
        } else if (clsConf) {
            qualityScore.textContent = `${clsConf}%`;
        }
    }
}

// Display classification results
function displayClassificationResults(data, opts = {}) {
    if (classificationResults) {
        classificationResults.classList.remove('hidden');
    }
    if (digitizationResults && !opts.keepDigitizationVisible) {
        digitizationResults.classList.add('hidden');
    }
    
    const pred = data?.prediction;
    if (!pred) {
        return;
    }
    
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
function displayDigitizationResults(data, opts = {}) {
    if (classificationResults && !opts.keepClassificationVisible) {
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
            const rate = data.metadata.sampling_rate_hz ?? data.sampling_rate_hz;
            samplingRate.textContent = rate ? `${rate} Hz` : '--';
        }
        if (signalLength) {
            const length = data.metadata.signal_length ?? data.num_samples;
            signalLength.textContent = length ? `${length} samples` : '--';
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
    if (typeof Plotly === 'undefined') {
        throw new Error('Plot library not loaded. Please refresh and try again.');
    }

    const leadNames = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];
    const colors = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6', '#ec4899',
                    '#06b6d4', '#84cc16', '#f97316', '#14b8a6', '#a855f7', '#eab308'];
    
    // Create container for plots
    const signalChart = document.getElementById('signalChart');
    if (!signalChart) return;
    
    signalChart.innerHTML = '<div id="ecgPlot" class="w-full" style="height: 900px;"></div>';

    const getLeadSignal = (lead) => {
        if (Array.isArray(signals[lead])) return signals[lead];
        const foundKey = Object.keys(signals).find((k) => k.toLowerCase() === lead.toLowerCase());
        return foundKey ? signals[foundKey] : null;
    };

    const availableLeadData = leadNames
        .map((lead) => ({ lead, data: getLeadSignal(lead) }))
        .filter((entry) => Array.isArray(entry.data) && entry.data.length > 0);

    if (availableLeadData.length === 0) {
        throw new Error('No ECG lead data found in response');
    }

    // Compute a stable vertical spacing so all leads are visible.
    let maxAbs = 0;
    availableLeadData.forEach((entry) => {
        entry.data.forEach((v) => {
            const n = Number(v);
            if (Number.isFinite(n)) {
                maxAbs = Math.max(maxAbs, Math.abs(n));
            }
        });
    });
    const offsetStep = Math.max(maxAbs * 3, 2);

    const traces = [];
    const annotations = [];

    availableLeadData.forEach((entry, idx) => {
        const leadIndex = leadNames.indexOf(entry.lead);
        const color = colors[Math.max(0, leadIndex) % colors.length];
        const signal = entry.data.map((v) => Number(v) || 0);
        const x = Array.from({ length: signal.length }, (_, i) => i);

        // Keep lead I at top and V6 at bottom.
        const verticalOffset = (availableLeadData.length - 1 - idx) * offsetStep;
        const y = signal.map((v) => v + verticalOffset);

        traces.push({
            x,
            y,
            type: 'scatter',
            mode: 'lines',
            name: `Lead ${entry.lead}`,
            line: {
                color,
                width: 1.2
            },
            hovertemplate: `Lead ${entry.lead}<br>Sample %{x}<br>Value %{y:.3f}<extra></extra>`
        });

        annotations.push({
            x: 0,
            y: verticalOffset,
            xref: 'x',
            yref: 'y',
            text: entry.lead,
            showarrow: false,
            xanchor: 'right',
            align: 'right',
            xshift: -8,
            font: {
                color,
                size: 11
            }
        });
    });

    const layout = {
        title: {
            text: `12-Lead ECG Signals (${availableLeadData.length} leads)`,
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
        xaxis: {
            title: { text: 'Sample', font: { color: '#999' } },
            gridcolor: '#1a1a1a',
            color: '#666',
            zeroline: false
        },
        yaxis: {
            title: { text: 'Amplitude (stacked)', font: { color: '#999' } },
            gridcolor: '#141414',
            color: '#666',
            showticklabels: false,
            zeroline: false
        },
        annotations,
        margin: { t: 60, b: 100, l: 90, r: 20 }
    };
    
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
    newUpload.addEventListener('click', (e) => {
        e.preventDefault();
        currentFile = null;
        currentResults = null;
        isProcessing = false;
        if (processBtn) {
            processBtn.disabled = false;
        }
        preview.classList.remove('hidden');
        results.classList.add('hidden');
        loading.classList.add('hidden');
        fileInput.value = '';
    });
}
