"""
Unified Inference Engine for ECG Analysis
Supports both classification and digitization tasks
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Tuple
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class UnifiedECGInferenceEngine:
    """
    Unified inference engine supporting multiple ECG tasks
    """
    
    LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    SAMPLING_RATE = 500  # Hz
    SIGNAL_DURATION = 2.0  # seconds
    CLASS_NAMES = ['Normal', 'Abnormal Heartbeat', 'Myocardial Infarction']
    
    def __init__(self, model_manager, preprocessor):
        self.model_manager = model_manager
        self.preprocessor = preprocessor
        logger.info("Unified inference engine initialized")
    
    def process_ecg_image(
        self,
        image_path: str,
        request_id: str,
        task: str = 'auto',
        options: Dict = None
    ) -> Dict:
        """
        Process ECG image for classification or digitization
        
        Args:
            image_path: Path to ECG image
            request_id: Unique request identifier
            task: 'classification', 'digitization', or 'auto'
            options: Processing options
        
        Returns:
            result: Dict containing predictions, metadata, and quality metrics
        """
        start_time = time.time()
        
        try:
            # Determine task
            if task == 'auto':
                available_models = self.model_manager.get_available_models()
                if 'digitization' in available_models:
                    task = 'digitization'
                elif 'classification' in available_models:
                    task = 'classification'
                else:
                    raise ValueError("No models available")
            
            logger.info(f"[{request_id}] Task: {task}")
            
            # 1. Preprocess image
            logger.info(f"[{request_id}] Preprocessing image...")
            input_tensor, preprocess_metadata = self.preprocessor.preprocess(image_path, options)
            
            # 2. Run inference
            logger.info(f"[{request_id}] Running inference...")
            predictions, model_type = self.model_manager.predict(input_tensor, model_type=task)
            
            # 3. Post-process based on task
            logger.info(f"[{request_id}] Post-processing results...")
            
            if task == 'classification':
                result_data, quality_metrics = self._post_process_classification(predictions)
            elif task == 'digitization':
                result_data, quality_metrics = self._post_process_digitization(predictions)
            else:
                raise ValueError(f"Unknown task: {task}")
            
            # 4. Calculate processing time
            processing_time = time.time() - start_time
            
            # 5. Compile result
            result = {
                'request_id': request_id,
                'task': task,
                **result_data,
                'metadata': {
                    'image_path': image_path,
                    'processing_time_seconds': round(processing_time, 3),
                    'timestamp': datetime.utcnow().isoformat(),
                    'model_type': model_type,
                    'preprocessing': preprocess_metadata
                },
                'quality_metrics': quality_metrics,
                'status': 'success'
            }
            
            # Add task-specific metadata
            if task == 'classification':
                result['metadata']['num_classes'] = len(self.CLASS_NAMES)
            elif task == 'digitization':
                result['metadata']['sampling_rate_hz'] = self.SAMPLING_RATE
                result['metadata']['signal_duration_sec'] = self.SIGNAL_DURATION
                result['metadata']['num_leads'] = len(self.LEAD_NAMES)
                if 'signals' in result_data:
                    result['metadata']['signal_length'] = len(result_data['signals'][self.LEAD_NAMES[0]])
            
            logger.info(f"[{request_id}] Processing completed in {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"[{request_id}] Processing failed: {str(e)}")
            raise
    
    def _post_process_classification(self, predictions: torch.Tensor) -> Tuple[Dict, Dict]:
        """
        Post-process classification predictions
        
        Args:
            predictions: Tensor of shape (B, num_classes) - raw logits
        
        Returns:
            classification_results, quality_metrics
        """
        import torch.nn.functional as F
        
        # Apply softmax
        probabilities = F.softmax(predictions, dim=1)
        
        # Get predicted class
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        predicted_class = self.CLASS_NAMES[predicted_class_idx]
        confidence = probabilities[0, predicted_class_idx].item()
        
        # Probability distribution
        prob_dist = {}
        for i, class_name in enumerate(self.CLASS_NAMES):
            prob_dist[class_name] = float(probabilities[0, i].item())
        
        classification_results = {
            'prediction': {
                'class': predicted_class,
                'class_index': predicted_class_idx,
                'confidence': round(confidence * 100, 2),
                'probability_distribution': prob_dist
            }
        }
        
        quality_metrics = {
            'confidence_score': round(confidence * 100, 2),
            'prediction_certainty': 'high' if confidence > 0.8 else ('medium' if confidence > 0.5 else 'low'),
            'entropy': float(-torch.sum(probabilities * torch.log(probabilities + 1e-10)).item())
        }
        
        return classification_results, quality_metrics
    
    def _post_process_digitization(self, predictions: torch.Tensor) -> Tuple[Dict, Dict]:
        """
        Post-process digitization predictions
        
        Args:
            predictions: Tensor of shape (B, 12, 1000) - reconstructed signals
        
        Returns:
            digitization_results, quality_metrics
        """
        # Convert to numpy
        signals_np = predictions[0].cpu().numpy()  # Shape: (12, 1000)
        
        # Denormalize signals to mV scale
        signals_np = self._denormalize_signals(signals_np)
        
        # Create dictionary mapping lead names to signals
        signals_dict = {}
        for i, lead_name in enumerate(self.LEAD_NAMES):
            signals_dict[lead_name] = signals_np[i].tolist()
        
        digitization_results = {
            'signals': signals_dict,
            'format': 'time_series',
            'leads': self.LEAD_NAMES,
            'sampling_rate_hz': self.SAMPLING_RATE,
            'duration_seconds': self.SIGNAL_DURATION,
            'num_samples': signals_np.shape[1]
        }
        
        # Calculate signal quality metrics
        quality_metrics = self._calculate_signal_quality(signals_np)
        
        return digitization_results, quality_metrics
    
    def _denormalize_signals(self, signals: np.ndarray) -> np.ndarray:
        """
        Denormalize signals from [-1, 1] to mV scale
        
        Args:
            signals: (12, 1000) normalized signals
        
        Returns:
            denormalized signals in mV
        """
        # Typical ECG amplitude ranges (in mV)
        amplitude_scales = {
            'I': 1.5, 'II': 1.5, 'III': 1.5,
            'aVR': 1.5, 'aVL': 1.5, 'aVF': 1.5,
            'V1': 3.0, 'V2': 3.0, 'V3': 3.0,
            'V4': 3.0, 'V5': 3.0, 'V6': 3.0
        }
        
        denormalized = np.zeros_like(signals)
        
        for i, lead_name in enumerate(self.LEAD_NAMES):
            lead_signal = signals[i]
            scale = amplitude_scales.get(lead_name, 2.0)
            
            # Signals are in range [-1, 1], scale to mV
            denormalized[i] = lead_signal * scale
        
        return denormalized
    
    def _calculate_signal_quality(self, signals: np.ndarray) -> Dict:
        """
        Calculate signal quality metrics
        
        Args:
            signals: (12, 1000) numpy array
        
        Returns:
            quality metrics dict
        """
        # Basic quality indicators
        metrics = {}
        
        # Signal-to-noise ratio (simplified)
        for i, lead_name in enumerate(self.LEAD_NAMES):
            signal = signals[i]
            
            # Calculate SNR approximation
            signal_power = np.mean(signal ** 2)
            noise_power = np.var(np.diff(signal))  # High-frequency variation as noise
            
            if noise_power > 1e-10:
                snr = 10 * np.log10(signal_power / noise_power)
            else:
                snr = 100.0  # Very high SNR
            
            metrics[f'snr_{lead_name}_db'] = round(float(snr), 2)
        
        # Overall quality score
        avg_snr = np.mean([v for k, v in metrics.items() if 'snr_' in k])
        
        if avg_snr > 20:
            quality_rating = 'excellent'
        elif avg_snr > 15:
            quality_rating = 'good'
        elif avg_snr > 10:
            quality_rating = 'acceptable'
        else:
            quality_rating = 'poor'
        
        metrics['overall_quality'] = quality_rating
        metrics['average_snr_db'] = round(avg_snr, 2)
        
        # Check for flat signals (model failure indicator)
        flat_leads = []
        for i, lead_name in enumerate(self.LEAD_NAMES):
            if np.std(signals[i]) < 0.01:
                flat_leads.append(lead_name)
        
        if flat_leads:
            metrics['warning'] = f"Potentially flat signals detected in leads: {', '.join(flat_leads)}"
        
        return metrics
    
    def export_signals(
        self,
        signals_dict: Dict[str, List[float]],
        output_path: str,
        format: str = 'csv'
    ) -> str:
        """
        Export signals to file
        
        Args:
            signals_dict: Dictionary of lead signals
            output_path: Output file path
            format: 'csv', 'json', or 'wfdb'
        
        Returns:
            path: Path to exported file
        """
        import pandas as pd
        import json
        from pathlib import Path
        
        output_path = Path(output_path)
        
        if format == 'csv':
            # Export as CSV
            df = pd.DataFrame(signals_dict)
            csv_path = output_path.with_suffix('.csv')
            df.to_csv(csv_path, index=False)
            logger.info(f"Signals exported to CSV: {csv_path}")
            return str(csv_path)
        
        elif format == 'json':
            # Export as JSON
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump({
                    'signals': signals_dict,
                    'metadata': {
                        'leads': self.LEAD_NAMES,
                        'sampling_rate_hz': self.SAMPLING_RATE,
                        'duration_seconds': self.SIGNAL_DURATION
                    }
                }, f, indent=2)
            logger.info(f"Signals exported to JSON: {json_path}")
            return str(json_path)
        
        elif format == 'wfdb':
            # Export in WFDB format
            try:
                import wfdb
                
                # Convert to numpy array
                signal_array = np.array([signals_dict[lead] for lead in self.LEAD_NAMES]).T
                
                # Create WFDB record
                record = wfdb.Record(
                    record_name=output_path.stem,
                    p_signal=signal_array,
                    sig_name=self.LEAD_NAMES,
                    fs=self.SAMPLING_RATE,
                    units=['mV'] * 12
                )
                
                wfdb_path = output_path.with_suffix('')
                wfdb.wrsamp(str(wfdb_path), record)
                
                logger.info(f"Signals exported to WFDB: {wfdb_path}")
                return str(wfdb_path)
            except ImportError:
                logger.error("wfdb package not installed")
                raise ValueError("WFDB export requires 'wfdb' package")
        
        else:
            raise ValueError(f"Unsupported format: {format}")


if __name__ == "__main__":
    print("Unified ECG Inference Engine")
    print("Supports: Classification and Digitization")
