"""
Image preprocessing pipeline for ECG digitization
Handles real-world artifacts: grid lines, noise, skew, rotation, illumination

CRITICAL: This preprocessing MUST match the training preprocessing exactly!
During training, minimal preprocessing was applied (only resize + normalize).
Aggressive preprocessing like grid removal, denoising, or illumination normalization
will destroy the signal features the model learned to recognize, causing flat outputs.
"""

import cv2
import numpy as np
from PIL import Image
import logging
from typing import Tuple, Dict, Optional
import torch
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


class ECGImagePreprocessor:
    """
    Robust preprocessing pipeline for ECG images
    Handles various acquisition artifacts
    """
    
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size
        
        # ImageNet normalization (same as training)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Custom to_tensor to avoid numpy integration issues
        # self.to_tensor = transforms.ToTensor()
    
    def _pil_to_tensor(self, pic):
        """
        Custom PIL to tensor conversion that avoids numpy integration issues
        Workaround for NumPy 2.x compatibility with PyTorch
        """
        # Convert PIL image to numpy array
        if pic.mode == 'RGB':
            img_array = np.array(pic, dtype=np.float32)
        else:
            img_array = np.array(pic.convert('RGB'), dtype=np.float32)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Convert HWC to CHW format
        img_array = img_array.transpose(2, 0, 1)
        
        # Convert to tensor via list (workaround for numpy integration issue)
        tensor = torch.tensor(img_array.tolist(), dtype=torch.float32)
        
        return tensor
    
    def preprocess(self, image_path: str, options: Dict = None) -> Tuple[torch.Tensor, Dict]:
        """
        Main preprocessing pipeline
        
        Args:
            image_path: Path to ECG image
            options: Processing options
                - remove_grid: bool
                - denoise: bool
                - correct_rotation: bool
                - correct_skew: bool
        
        Returns:
            processed_tensor: torch.Tensor (1, 3, 256, 256)
            metadata: Dict with processing information
        """
        options = options or {}
        metadata = {
            'preprocessing_steps': [],
            'image_quality': {},
            'transformations': []
        }
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_shape = image.shape
            metadata['original_shape'] = original_shape
            
            # CRITICAL: Match training preprocessing exactly
            # During training, only simple resizing was done - NO aggressive preprocessing
            
            # Optional: Only apply minimal corrections if explicitly requested
            # These are disabled by default to match training
            if options.get('correct_rotation', False):  # Changed default to False
                image, angle = self._correct_rotation(image)
                if abs(angle) > 1:
                    metadata['preprocessing_steps'].append('rotation_correction')
                    metadata['transformations'].append({'rotation': angle})
            
            if options.get('correct_skew', False):  # Changed default to False
                image = self._correct_skew(image)
                metadata['preprocessing_steps'].append('skew_correction')
            
            # REMOVED: Illumination normalization (not in training)
            # REMOVED: Grid line removal (not in training)
            # REMOVED: Denoising (not in training)
            
            # Resize to target size (same as training)
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
            metadata['final_shape'] = image.shape
            metadata['preprocessing_steps'].append('resize')
            
            # Convert to tensor and normalize (EXACTLY as in training)
            # Training: image = image.astype(np.float32) / 255.0
            #           image = (image - mean) / std
            image_pil = Image.fromarray(image)
            tensor = self._pil_to_tensor(image_pil)  # Already divides by 255
            tensor = self.normalize(tensor)  # Apply ImageNet normalization
            tensor = tensor.unsqueeze(0)  # Add batch dimension
            
            metadata['preprocessing_steps'].append('normalization')
            
            return tensor, metadata
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise
    
    def _correct_rotation(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detect and correct image rotation"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
            
            if lines is not None and len(lines) > 0:
                # Calculate average angle
                angles = []
                for line in lines[:min(10, len(lines))]:
                    rho, theta = line[0]
                    angle = theta * 180 / np.pi
                    # Normalize to [-90, 90]
                    if angle > 90:
                        angle -= 180
                    angles.append(angle)
                
                # Use median angle
                rotation_angle = np.median(angles)
                
                # Only correct if significant
                if abs(rotation_angle) > 1:
                    # Rotate image
                    h, w = image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                    rotated = cv2.warpAffine(image, M, (w, h), 
                                           flags=cv2.INTER_CUBIC,
                                           borderMode=cv2.BORDER_REPLICATE)
                    return rotated, rotation_angle
            
            return image, 0.0
            
        except Exception as e:
            logger.warning(f"Rotation correction failed: {str(e)}")
            return image, 0.0
    
    def _correct_skew(self, image: np.ndarray) -> np.ndarray:
        """Correct perspective skew"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Find edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour (likely the ECG paper boundary)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Approximate to polygon
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # If we have a quadrilateral, apply perspective transform
                if len(approx) == 4:
                    pts = approx.reshape(4, 2)
                    
                    # Order points: top-left, top-right, bottom-right, bottom-left
                    rect = self._order_points(pts)
                    
                    # Compute perspective transform
                    h, w = image.shape[:2]
                    dst = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
                    
                    M = cv2.getPerspectiveTransform(rect.astype(np.float32), dst)
                    warped = cv2.warpPerspective(image, M, (w, h))
                    
                    return warped
            
            return image
            
        except Exception as e:
            logger.warning(f"Skew correction failed: {str(e)}")
            return image
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points in clockwise order starting from top-left"""
        # Sort by y-coordinate
        y_sorted = pts[np.argsort(pts[:, 1])]
        
        # Top two points
        top = y_sorted[:2]
        top = top[np.argsort(top[:, 0])]
        
        # Bottom two points
        bottom = y_sorted[2:]
        bottom = bottom[np.argsort(bottom[:, 0])]
        
        return np.array([top[0], top[1], bottom[1], bottom[0]])
    
    def _normalize_illumination(self, image: np.ndarray) -> np.ndarray:
        """Normalize illumination using CLAHE"""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_equalized = clahe.apply(l)
            
            # Merge and convert back
            lab_equalized = cv2.merge([l_equalized, a, b])
            normalized = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2RGB)
            
            return normalized
            
        except Exception as e:
            logger.warning(f"Illumination normalization failed: {str(e)}")
            return image
    
    def _remove_grid_lines(self, image: np.ndarray) -> np.ndarray:
        """Remove ECG grid lines using morphological operations"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply binary threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Detect horizontal lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
            
            # Detect vertical lines
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
            
            # Combine grid lines
            grid = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)
            
            # Remove grid from original
            result = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(grid))
            
            # Fill removed areas with white
            result[grid > 0] = 255
            
            return result
            
        except Exception as e:
            logger.warning(f"Grid removal failed: {str(e)}")
            return image
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising"""
        try:
            # Use Non-local Means Denoising
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            return denoised
        except Exception as e:
            logger.warning(f"Denoising failed: {str(e)}")
            return image
    
    def validate_image(self, image_path: str) -> Dict:
        """
        Validate image quality and suitability for processing
        
        Returns:
            validation_result: Dict with validation metrics
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'valid': False,
                    'quality_score': 0.0,
                    'issues': ['Failed to load image'],
                    'recommendations': ['Check file format and integrity']
                }
            
            h, w = image.shape[:2]
            issues = []
            recommendations = []
            
            # Check dimensions
            if w < 200 or h < 200:
                issues.append('Image resolution too low')
                recommendations.append('Use higher resolution image (min 200x200)')
            
            # Check aspect ratio
            aspect_ratio = w / h
            if aspect_ratio < 0.5 or aspect_ratio > 3:
                issues.append('Unusual aspect ratio')
                recommendations.append('Ensure ECG image is properly framed')
            
            # Check brightness
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            if mean_brightness < 50:
                issues.append('Image too dark')
                recommendations.append('Improve lighting or scan quality')
            elif mean_brightness > 200:
                issues.append('Image overexposed')
                recommendations.append('Reduce exposure or brightness')
            
            # Check contrast
            std_brightness = np.std(gray)
            if std_brightness < 20:
                issues.append('Low contrast')
                recommendations.append('Increase contrast or use better source')
            
            # Calculate quality score
            quality_score = 1.0
            quality_score -= 0.2 * len(issues)
            quality_score = max(0.0, min(1.0, quality_score))
            
            return {
                'valid': len(issues) == 0 or quality_score >= 0.3,
                'quality_score': quality_score,
                'dimensions': {'width': w, 'height': h},
                'aspect_ratio': aspect_ratio,
                'mean_brightness': float(mean_brightness),
                'contrast_std': float(std_brightness),
                'issues': issues,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return {
                'valid': False,
                'quality_score': 0.0,
                'issues': [f'Validation error: {str(e)}'],
                'recommendations': ['Check file format and integrity']
            }
