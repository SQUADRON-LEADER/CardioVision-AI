"""
Quick integration test for CardioVision AI backend
"""
import sys
import os
sys.path.insert(0, 'flask_backend')

from model_loader import ModelManager, HybridECGNet
from preprocessing import ECGImagePreprocessor
from inference import ECGInferenceEngine
from config import Config
import glob

print("=" * 70)
print("CardioVision AI - Integration Test")
print("=" * 70)

# Initialize components
print("\n1. Initializing components...")
model_manager = ModelManager(Config.MODEL_PATH)
preprocessor = ECGImagePreprocessor(target_size=Config.IMAGE_SIZE)
inference_engine = ECGInferenceEngine(model_manager, preprocessor)

# Load model
print("\n2. Loading model...")
try:
    model_manager.load_model()
    print(f"   ✓ Model loaded: {model_manager.model_metadata.get('architecture')}")
    print(f"   ✓ Number of classes: {model_manager.model_metadata.get('num_classes')}")
    print(f"   ✓ Test accuracy: {model_manager.model_metadata.get('test_accuracy', 'N/A')}")
except Exception as e:
    print(f"   ✗ Failed to load model: {e}")
    sys.exit(1)

# Find a test image
print("\n3. Looking for test ECG images...")
test_dirs = [
    'ecg_data/normal_ecg_images',
    'ecg_data/abnormal_heartbeat_ecg_images',
    'ecg_data/myocardial_infarction_ecg_images'
]

test_image = None
for test_dir in test_dirs:
    if os.path.exists(test_dir):
        images = glob.glob(f"{test_dir}/*.png") + glob.glob(f"{test_dir}/*.jpg")
        if images:
            test_image = images[0]
            print(f"   ✓ Found test image: {test_image}")
            break

if not test_image:
    print("   ✗ No test images found")
    print("   Please make sure ECG images exist in ecg_data/ directories")
    sys.exit(1)

# Process image
print("\n4. Processing test image...")
try:
    result = inference_engine.process_ecg_image(
        image_path=test_image,
        request_id="test-001",
        options={}
    )
    
    print(f"   ✓ Processing completed in {result['metadata']['processing_time_seconds']:.3f}s")
    
    if result.get('prediction'):
        pred = result['prediction']
        print(f"\n   Classification Result:")
        print(f"   - Predicted Class: {pred['class']}")
        print(f"   - Confidence: {pred['confidence']}%")
        print(f"   - Probability Distribution:")
        for class_name, prob in pred['probability_distribution'].items():
            print(f"     * {class_name}: {prob*100:.2f}%")
    
    print(f"\n   Quality Metrics:")
    print(f"   - Confidence Score: {result['quality_metrics'].get('confidence_score', 'N/A')}")
    print(f"   - Prediction Certainty: {result['quality_metrics'].get('prediction_certainty', 'N/A')}")
    
except Exception as e:
    import traceback
    print(f"   ✗ Processing failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED!")
print("=" * 70)
print("\nThe backend is ready to use. Start it with:")
print("  cd flask_backend && python app.py")
print("\nOr use the quick start script:")
print("  .\\start_backend.ps1")
