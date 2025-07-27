#!/usr/bin/env python3
"""
Pneumonia Detection System - Standalone Predictor (Keras Format)
Trained with 92.87% accuracy using 5-fold cross-validation
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict

# TensorFlow / Keras import
try:
    import tensorflow as tf
    from tensorflow.keras.utils import load_img, img_to_array
    from keras.models import load_model as keras_load_model  # NEW loader
    print(f"TensorFlow {tf.__version__} loaded successfully")
except ImportError:
    print("ERROR: TensorFlow not installed. Please install with: pip install tensorflow")
    sys.exit(1)

class PneumoniaEnsemblePredictor:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.models = []
        self.load_models()

    def load_models(self):
        print("🔍 Loading Keras-format ensemble models...")

        for i in range(1, 6):
            model_path = self.model_dir / f"pneumonia_mobilenet_fold_{i}.keras"
            if model_path.exists():
                try:
                    model = keras_load_model(model_path)  # use Keras native
                    self.models.append(model)
                    print(f"  ✅ Loaded fold {i}")
                except Exception as e:
                    print(f"  ❌ Failed to load fold {i}: {e}")
            else:
                print(f"  ❌ Model not found: {model_path}")

        if not self.models:
            raise RuntimeError("❌ No models were loaded. Check paths and file formats.")

        print(f"✅ Successfully loaded {len(self.models)} models.")

    def preprocess_image(self, image_path: str) -> np.ndarray:
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        return np.expand_dims(img_array, axis=0)

    def predict_single(self, image_path: str) -> Dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img_batch = self.preprocess_image(image_path)

        predictions = [float(model.predict(img_batch, verbose=0)[0][0]) for model in self.models]
        ensemble_pred = np.mean(predictions)
        ensemble_std = np.std(predictions)
        confidence = ensemble_pred if ensemble_pred > 0.5 else 1 - ensemble_pred

        if ensemble_std < 0.1 and confidence > 0.9:
            quality = "Very High"
        elif ensemble_std < 0.15 and confidence > 0.8:
            quality = "High"
        elif ensemble_std < 0.2 and confidence > 0.7:
            quality = "Moderate"
        else:
            quality = "Low"

        clinical_action = (
            "High confidence - consider immediate clinical review" if confidence > 0.9 else
            "Moderate confidence - clinical correlation recommended" if confidence > 0.8 else
            "Low confidence - additional imaging or clinical assessment needed"
        )

        return {
            'image_path': image_path,
            'prediction': 'PNEUMONIA' if ensemble_pred > 0.5 else 'NORMAL',
            'probability': float(ensemble_pred),
            'confidence': float(confidence * 100),
            'uncertainty': float(ensemble_std),
            'prediction_quality': quality,
            'clinical_action': clinical_action,
            'individual_predictions': predictions,
            'models_used': len(self.models)
        }

    def predict_folder(self, folder_path: str, output_file: str = None) -> List[Dict]:
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in folder.iterdir() if f.suffix.lower() in exts]

        print(f"\n📁 Found {len(image_files)} images")
        results = []

        for i, img_path in enumerate(image_files, 1):
            try:
                result = self.predict_single(str(img_path))
                results.append(result)
                print(f"[{i}/{len(image_files)}] {img_path.name}: {result['prediction']} ({result['confidence']:.1f}%)")
            except Exception as e:
                print(f"[{i}/{len(image_files)}] Error: {e}")

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n📄 Results saved to: {output_file}")

        pn = sum(1 for r in results if r['prediction'] == 'PNEUMONIA')
        nm = len(results) - pn
        print(f"\nSummary:")
        print(f"  Pneumonia: {pn} ({pn/len(results)*100:.1f}%)")
        print(f"  Normal: {nm} ({nm/len(results)*100:.1f}%)")

        return results


def main():
    parser = argparse.ArgumentParser(description="Pneumonia Detector (Keras Format)")
    parser.add_argument("input", help="Image or folder path")
    parser.add_argument("--model-dir", default=".", help="Folder with .keras models")
    parser.add_argument("--output", "-o", help="JSON file for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose mode")

    args = parser.parse_args()
    predictor = PneumoniaEnsemblePredictor(args.model_dir)

    input_path = Path(args.input)
    if input_path.is_file():
        result = predictor.predict_single(str(input_path))
        if args.verbose:
            print(json.dumps(result, indent=2))
        else:
            print(f"\nPrediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.1f}%")
            print(f"Action: {result['clinical_action']}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump([result], f, indent=2)
            print(f"\nResult saved to: {args.output}")

    elif input_path.is_dir():
        predictor.predict_folder(str(input_path), args.output)
    else:
        print("❌ Invalid input path")
        sys.exit(1)

if __name__ == "__main__":
    main()
