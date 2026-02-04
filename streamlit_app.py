"""
ECG Digitization Streamlit Application
Frontend and Backend in one application
"""

import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import io
import tempfile
import os
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import traceback

# Dummy class for unpickling notebook-saved models
class ECGDigitizationConfig:
    """Dummy config class for unpickling training checkpoint"""
    pass

# Make the class available in __main__ for unpickling
sys.modules['__main__'].ECGDigitizationConfig = ECGDigitizationConfig

# Set page config
st.set_page_config(
    page_title="CardioVision AI - ECG Digitization",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Model Architecture (Must match training)
# ============================================================================

class SimpleCNN(nn.Module):
    """Simplified CNN backbone for feature extraction"""
    
    def __init__(self, input_channels=3):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1 - stride 2
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),  # 128x128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64
            
            # Block 2 - stride 2  
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16
            
            # Block 3 - stride 2
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4
            
            # Block 4 - stride 1
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # 4x4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.features(x)


class ECGDigitizationModel(nn.Module):
    """ECG Digitization Model - Must match training architecture"""
    
    def __init__(self, num_leads=12, signal_length=1000, lstm_hidden_dim=128, lstm_layers=1):
        super().__init__()
        self.num_leads = num_leads
        self.signal_length = signal_length
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        
        # Vision backbone - simplified CNN
        self.backbone = SimpleCNN(input_channels=3)
        backbone_channels = 512
        
        # Feature compression
        self.feature_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_channels * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Sequence decoder - BiLSTM for temporal modeling
        self.sequence_decoder = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if lstm_layers > 1 else 0
        )
        
        # Multi-lead output projection
        lstm_output_size = lstm_hidden_dim * 2  # Bidirectional
        self.lead_projectors = nn.ModuleList([
            nn.Linear(lstm_output_size, 1) for _ in range(num_leads)
        ])
        
        # Lead attention mechanism
        self.lead_attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Final projection layer (as in training)
        self.final_projection = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_leads)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract visual features
        visual_features = self.backbone(x)  # [B, 512, 4, 4]
        
        # Project to sequence format
        projected_features = self.feature_projection(visual_features)  # [B, 512]
        
        # Expand for sequence generation
        sequence_input = projected_features.unsqueeze(1).expand(
            batch_size, self.signal_length, -1
        )  # [B, signal_length, 512]
        
        # Generate sequence with LSTM
        lstm_output, _ = self.sequence_decoder(sequence_input)  # [B, signal_length, lstm_hidden*2]
        
        # Apply attention
        attended_output, _ = self.lead_attention(lstm_output, lstm_output, lstm_output)
        
        # Generate multi-lead output
        output = self.final_projection(attended_output)  # [B, signal_length, num_leads]
        
        # Transpose to match target format [B, num_leads, signal_length]
        output = output.transpose(1, 2)
        
        return output


# ============================================================================
# Preprocessing
# ============================================================================

def preprocess_image(image, target_size=(256, 256)):
    """
    Preprocess ECG image for inference
    Matches training preprocessing exactly
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Resize
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Convert to float and normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    
    # Convert to tensor (CHW format)
    image = image.transpose(2, 0, 1)
    tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    
    return tensor


# ============================================================================
# Model Loading and Caching
# ============================================================================

@st.cache_resource
def load_model(model_path):
    """Load the trained model (cached)"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        model = ECGDigitizationModel(
            num_leads=12,
            signal_length=1000,
            lstm_hidden_dim=128,
            lstm_layers=1
        )
        
        # Load checkpoint (weights_only=False for models with numpy objects)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Extract state dict (handle different checkpoint formats)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load state dict with strict=False to handle architecture variations
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            st.warning(f"Missing keys in model: {missing_keys[:5]}...")  # Show first 5
        if unexpected_keys:
            st.info(f"Unexpected keys (ignored): {unexpected_keys[:5]}...")  # Show first 5
        
        model.to(device)
        model.eval()
        
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error(traceback.format_exc())
        return None, None


# ============================================================================
# Inference
# ============================================================================

def perform_inference(model, device, image_tensor):
    """Run inference on preprocessed image"""
    try:
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            predictions = model(image_tensor)
            signals = predictions.cpu().numpy()[0]  # Shape: (12, 1000)
            
            # Debug information
            st.sidebar.markdown("### 🔍 Debug Info")
            st.sidebar.text(f"Output shape: {signals.shape}")
            st.sidebar.text(f"Value range: [{signals.min():.3f}, {signals.max():.3f}]")
            st.sidebar.text(f"Mean: {signals.mean():.3f}")
            st.sidebar.text(f"Std: {signals.std():.3f}")
            
            # Check if output looks valid
            if abs(signals.std()) < 0.001:
                st.warning("⚠️ Warning: Output signals have very low variance. Model may need retraining.")
            
        return signals
    except Exception as e:
        st.error(f"Error during inference: {str(e)}")
        st.error(traceback.format_exc())
        return None


# ============================================================================
# Visualization
# ============================================================================

def plot_ecg_signals(signals, lead_names=None, sampling_rate=500):
    """Create interactive plotly visualization of ECG signals"""
    if lead_names is None:
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    num_leads = signals.shape[0]
    time_axis = np.arange(signals.shape[1]) / sampling_rate
    
    # Normalize each lead for better visualization
    normalized_signals = []
    for i in range(num_leads):
        signal = signals[i]
        # Standardize each lead
        signal_mean = signal.mean()
        signal_std = signal.std()
        if signal_std > 0:
            normalized = (signal - signal_mean) / signal_std
        else:
            normalized = signal - signal_mean
        normalized_signals.append(normalized)
    
    normalized_signals = np.array(normalized_signals)
    
    # Create subplot figure
    fig = go.Figure()
    
    # Add traces for each lead with vertical offset
    spacing = 3.0  # Vertical spacing between leads
    for i, lead_name in enumerate(lead_names[:num_leads]):
        offset = (num_leads - i - 1) * spacing  # Vertical spacing
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=normalized_signals[i] + offset,
            mode='lines',
            name=lead_name,
            line=dict(width=1.5, color=f'hsl({i * 30}, 70%, 50%)'),
            hovertemplate=f'{lead_name}<br>Time: %{{x:.3f}}s<br>Value: %{{customdata:.3f}}<extra></extra>',
            customdata=signals[i]  # Show original values in hover
        ))
        
        # Add lead label on the left
        fig.add_annotation(
            x=-0.02,
            y=offset,
            text=f"<b>{lead_name}</b>",
            xref="paper",
            yref="y",
            showarrow=False,
            xanchor="right",
            font=dict(size=12, color=f'hsl({i * 30}, 70%, 50%)')
        )
    
    # Update layout
    fig.update_layout(
        title={
            'text': "ECG Signal Visualization - 12 Leads",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'white'}
        },
        xaxis_title="Time (seconds)",
        yaxis_title="",
        height=900,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        plot_bgcolor='rgb(17, 17, 17)',
        paper_bgcolor='rgb(17, 17, 17)',
        font=dict(color='white'),
        yaxis=dict(
            showticklabels=False,
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zeroline=False
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zeroline=True,
            zerolinecolor='rgba(255, 255, 255, 0.3)'
        ),
        margin=dict(l=80, r=40, t=80, b=60)
    )
    
    return fig


def create_download_csv(signals, lead_names=None, sampling_rate=500):
    """Create CSV file for download"""
    if lead_names is None:
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # Create dataframe
    time_axis = np.arange(signals.shape[1]) / sampling_rate
    df = pd.DataFrame(signals.T, columns=lead_names[:signals.shape[0]])
    df.insert(0, 'Time(s)', time_axis)
    
    return df


# ============================================================================
# Main Application
# ============================================================================

def main():
    st.title("🫀 CardioVision AI")
    st.markdown("### AI-Powered ECG Image to Signal Digitization")
    st.markdown("Convert ECG images to digital signals using advanced deep learning")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Use only the best trained model
        model_path = "checkpoints/best_model.pth"
        st.info("**Using:** Best Trained Model")
        
        st.markdown("---")
        st.markdown("### 📊 Information")
        st.info("""
        **Model Details:**
        - 12-lead ECG
        - 500 Hz sampling rate
        - 2 seconds duration
        - 1000 samples per lead
        """)
        
        st.markdown("---")
        st.markdown("### 📝 Instructions")
        st.markdown("""
        1. Upload an ECG image
        2. View the digitized signals
        3. Download as CSV
        """)
    
    # Main content
    st.markdown("---")
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.info("Please ensure the model file exists at the specified path.")
        return
    
    # Load model
    with st.spinner(f"Loading model from {model_path}..."):
        model, device = load_model(model_path)
    
    if model is None:
        st.error("Failed to load model. Please check the logs above.")
        return
    
    st.success(f"✅ Model loaded successfully on {device}")
    
    # File upload
    st.markdown("### 📤 Upload ECG Image")
    uploaded_file = st.file_uploader(
        "Choose an ECG image file",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload a clear image of a 12-lead ECG"
    )
    
    if uploaded_file is not None:
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🖼️ Input ECG Image")
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True, caption="Original ECG Image")
            
            # Show image details
            st.markdown(f"**Size:** {image.size[0]} x {image.size[1]} pixels")
            st.markdown(f"**Format:** {image.format}")
            st.markdown(f"**Mode:** {image.mode}")
            
            # Add info about expected input
            with st.expander("ℹ️ What makes a good ECG image?"):
                st.markdown("""
                - Clear 12-lead ECG printout
                - Visible grid lines and waveforms
                - Good contrast and lighting
                - Minimal blur or distortion
                - Standard ECG paper format
                """)
        
        with col2:
            st.markdown("### 🔄 Processing")
            
            # Process button
            if st.button("🚀 Digitize ECG", type="primary", use_container_width=True):
                try:
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Step 1: Preprocess
                    status_text.text("Preprocessing image...")
                    progress_bar.progress(25)
                    image_tensor = preprocess_image(image)
                    
                    # Step 2: Inference
                    status_text.text("Running inference...")
                    progress_bar.progress(50)
                    signals = perform_inference(model, device, image_tensor)
                    
                    if signals is None:
                        st.error("Inference failed!")
                        return
                    
                    # Step 3: Post-process
                    status_text.text("Post-processing results...")
                    progress_bar.progress(75)
                    
                    # Store in session state
                    st.session_state['signals'] = signals
                    st.session_state['processed'] = True
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Complete!")
                    
                    st.success("🎉 ECG digitization completed successfully!")
                    
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
                    st.error(traceback.format_exc())
        
        # Display results if available
        if st.session_state.get('processed', False):
            signals = st.session_state['signals']
            
            st.markdown("---")
            st.markdown("### 📈 Digitized ECG Signals")
            
            # Define lead names
            lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            
            # Tabs for different views
            tab1, tab2, tab3 = st.tabs(["📊 All Leads", "🔍 Individual Leads", "📉 Sample Lead Detail"])
            
            with tab1:
                # Plot all signals
                fig = plot_ecg_signals(signals)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Individual lead plots (grid view)
                st.markdown("#### Individual Lead Waveforms")
                cols = st.columns(3)
                time_axis = np.arange(signals.shape[1]) / 500
                
                for i, lead_name in enumerate(lead_names):
                    with cols[i % 3]:
                        fig_single = go.Figure()
                        fig_single.add_trace(go.Scatter(
                            x=time_axis,
                            y=signals[i],
                            mode='lines',
                            name=lead_name,
                            line=dict(width=2, color=f'hsl({i * 30}, 70%, 50%)')
                        ))
                        fig_single.update_layout(
                            title=f"Lead {lead_name}",
                            xaxis_title="Time (s)",
                            yaxis_title="Amplitude",
                            height=200,
                            margin=dict(l=40, r=20, t=40, b=30),
                            plot_bgcolor='rgb(240, 240, 240)'
                        )
                        st.plotly_chart(fig_single, use_container_width=True, key=f"lead_{i}")
            
            with tab3:
                # Detailed view of one lead (e.g., Lead II - most diagnostic)
                st.markdown("#### Lead II - Detailed View")
                st.info("Lead II is typically the most diagnostic lead for rhythm analysis")
                
                lead_ii_signal = signals[1]  # Lead II
                time_axis = np.arange(len(lead_ii_signal)) / 500
                
                fig_detail = go.Figure()
                fig_detail.add_trace(go.Scatter(
                    x=time_axis,
                    y=lead_ii_signal,
                    mode='lines',
                    name='Lead II',
                    line=dict(width=2, color='red')
                ))
                
                # Add markers for potential peaks
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(lead_ii_signal, distance=50, prominence=0.1)
                
                if len(peaks) > 0:
                    fig_detail.add_trace(go.Scatter(
                        x=time_axis[peaks],
                        y=lead_ii_signal[peaks],
                        mode='markers',
                        name='Detected Peaks',
                        marker=dict(size=8, color='green', symbol='x')
                    ))
                    
                    # Calculate approximate heart rate
                    if len(peaks) > 1:
                        avg_rr_interval = np.mean(np.diff(peaks)) / 500  # in seconds
                        heart_rate = 60 / avg_rr_interval
                        st.metric("Estimated Heart Rate", f"{heart_rate:.0f} BPM")
                
                fig_detail.update_layout(
                    xaxis_title="Time (seconds)",
                    yaxis_title="Amplitude",
                    height=400,
                    hovermode='x unified',
                    plot_bgcolor='white',
                    xaxis=dict(showgrid=True, gridcolor='lightgray'),
                    yaxis=dict(showgrid=True, gridcolor='lightgray')
                )
                st.plotly_chart(fig_detail, use_container_width=True)
            
            # Statistics and Quality Check
            st.markdown("### 📊 Signal Statistics & Quality Check")
            lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            
            # Quality metrics
            col_q1, col_q2, col_q3, col_q4 = st.columns(4)
            
            overall_std = signals.std()
            overall_range = signals.max() - signals.min()
            lead_variations = [signals[i].std() for i in range(12)]
            avg_lead_std = np.mean(lead_variations)
            
            with col_q1:
                st.metric("Overall Std Dev", f"{overall_std:.3f}")
            with col_q2:
                st.metric("Signal Range", f"{overall_range:.3f}")
            with col_q3:
                st.metric("Avg Lead Std", f"{avg_lead_std:.3f}")
            with col_q4:
                # Quality indicator
                if overall_std > 0.1 and overall_range > 0.5:
                    st.metric("Quality", "✅ Good")
                elif overall_std > 0.01:
                    st.metric("Quality", "⚠️ Fair")
                else:
                    st.metric("Quality", "❌ Poor")
            
            # Quality assessment
            if overall_std < 0.01:
                st.error("⚠️ **Warning:** Signals have very low variation. This may indicate:")
                st.markdown("""
                - Model not properly loaded
                - Input image quality issues
                - Model needs retraining with better data
                """)
            elif overall_std > 0.1:
                st.success("✅ **Signal quality looks good!** The model has generated varied ECG waveforms.")
            else:
                st.warning("⚠️ **Moderate quality:** Signals show some variation but may need verification.")
            
            # Detailed statistics per lead
            with st.expander("📈 Detailed Lead Statistics"):
                stats_data = []
                for i, lead in enumerate(lead_names):
                    stats_data.append({
                        'Lead': lead,
                        'Mean': f"{signals[i].mean():.3f}",
                        'Std': f"{signals[i].std():.3f}",
                        'Min': f"{signals[i].min():.3f}",
                        'Max': f"{signals[i].max():.3f}",
                        'Range': f"{signals[i].max() - signals[i].min():.3f}"
                    })
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
            
            # Download section
            st.markdown("### 💾 Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV download
                csv_df = create_download_csv(signals)
                csv_buffer = io.StringIO()
                csv_df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv_data,
                    file_name=f"ecg_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # NumPy download
                numpy_buffer = io.BytesIO()
                np.save(numpy_buffer, signals)
                numpy_data = numpy_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download as NumPy",
                    data=numpy_data,
                    file_name=f"ecg_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy",
                    mime="application/octet-stream",
                    use_container_width=True
                )
    
    else:
        st.info("👆 Please upload an ECG image to get started")
    
    # Footer
    st.markdown("---")
    
    # Model information
    with st.expander("ℹ️ About the Model"):
        st.markdown("""
        ### Model Architecture
        - **Type:** CNN + BiLSTM with Attention
        - **Backbone:** SimpleCNN (4 convolutional blocks)
        - **Sequence Decoder:** Bidirectional LSTM (128 hidden units)
        - **Output:** 12 leads × 1000 samples (2 seconds @ 500 Hz)
        
        ### Training Details
        - **Dataset:** 80/20 train-validation split
        - **Loss Function:** Multi-component (MSE + Morphology + Temporal)
        - **Best Model:** Selected based on validation loss
        
        ### Expected Performance
        - **Correlation:** ~0.6-0.8 with ground truth
        - **RMSE:** Variable depending on ECG quality
        - **Signal Characteristics:** Should show clear P, QRS, T waves
        """)
    
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <p>ECG Digitization Service | Powered by PyTorch & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state['processed'] = False
    
    main()
