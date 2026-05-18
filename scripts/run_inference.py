#!/usr/bin/env python3
"""Simple CLI to run a single inference using the backend inference engine.

This tool is intentionally lightweight and intended for manual local use.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

# Make backend importable
ROOT = Path(__file__).resolve().parents[0].parent
sys.path.insert(0, str(ROOT / 'backend'))

from inference_unified import UnifiedECGInferenceEngine
from model_loader_unified import ModelManager as UnifiedModelManager
from preprocessing import ECGImagePreprocessor
from config import get_config_dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('run_inference')


def main():
    parser = argparse.ArgumentParser(description='Run a single ECG inference')
    parser.add_argument('image', help='Path to ECG image file')
    parser.add_argument('--task', choices=['auto', 'classification', 'digitization', 'pipeline'], default='auto')
    parser.add_argument('--output', help='Path to write JSON result', default=None)
    args = parser.parse_args()

    cfg = get_config_dict()
    preprocessor = ECGImagePreprocessor(target_size=cfg.get('IMAGE_SIZE', (224, 224)))

    # Initialize model manager and engine if possible
    try:
        model_manager = UnifiedModelManager(cfg.get('MODEL_PATHS'))
        engine = UnifiedECGInferenceEngine(model_manager, preprocessor)
    except Exception:
        logger.error('Unified model manager not available or failed to initialize')
        raise

    result = engine.process_ecg_image(str(args.image), request_id='cli-run', task=args.task, options={})
    out = json.dumps(result, indent=2)

    if args.output:
        Path(args.output).write_text(out)
        logger.info(f'Wrote result to: {args.output}')
    else:
        print(out)


if __name__ == '__main__':
    main()
