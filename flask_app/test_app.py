"""
Automated Integration and End-to-End Test Suite for Flask Application.
"""

import unittest
import json
from pathlib import Path
from PIL import Image
import io

from flask_app.app import create_app
from flask_app.config import Config
from flask_app.models.registry import MODEL_METADATA, CLASS_NAMES
from flask_app.models.model_loader import ModelCache, benchmark_model
from flask_app.models.predictor import predict_paddy_leaf


class FlaskAppTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()

    def test_01_homepage_renders(self):
        """Verify homepage / loads with status 200 and key indicators."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Paddy Leaf Disease Detection", resp.data)
        self.assertIn(b"10,407", resp.data)

    def test_02_predict_page_renders(self):
        """Verify /predict renders with model dropdown and samples."""
        resp = self.client.get("/predict")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Select Deep Learning Architecture", resp.data)
        self.assertIn(b"Instant Test Samples", resp.data)

    def test_03_performance_page_renders(self):
        """Verify /model-performance renders with table and charts."""
        resp = self.client.get("/model-performance")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Verified Test Set Performance Table", resp.data)
        self.assertIn(b"accuracyF1Chart", resp.data)

    def test_04_class_performance_page_renders(self):
        """Verify /class-performance renders 10 classes."""
        resp = self.client.get("/class-performance?model=resnet50")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Bacterial Leaf Blight", resp.data)
        self.assertIn(b"Tungro", resp.data)

    def test_05_confusion_matrix_page_renders(self):
        """Verify /confusion-matrix renders with image paths."""
        resp = self.client.get("/confusion-matrix?model=efficientnet_b0")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Normalized Confusion Matrix", resp.data)

    def test_06_model_selection_page_renders(self):
        """Verify /model-selection renders matrix and live benchmarking panel."""
        resp = self.client.get("/model-selection")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Live Server Hardware Benchmark Suite", resp.data)

    def test_07_about_page_renders(self):
        """Verify /about renders research documentation and disclaimer."""
        resp = self.client.get("/about")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"anthony2261/paddy-disease-classification", resp.data)
        self.assertIn(b"Agricultural & Diagnostic Disclaimer", resp.data)

    def test_08_api_samples_endpoint(self):
        """Verify /api/samples returns sample metadata and thumbnails."""
        resp = self.client.get("/api/samples")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"])
        self.assertEqual(len(data["samples"]), 10)

    def test_09_api_metrics_endpoint(self):
        """Verify /api/metrics returns test metrics."""
        resp = self.client.get("/api/metrics")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"])
        self.assertIn("resnet50", data["data"])
        self.assertAlmostEqual(data["data"]["resnet50"]["accuracy_pct"], 96.93, places=1)

    def test_10_all_models_can_load_and_infer(self):
        """Verify all 5 models can load, pass validation, and infer a sample."""
        sample_path = Config.SAMPLES_DIR / "sample_0_bacterial_leaf_blight.jpg"
        self.assertTrue(sample_path.exists())

        for model_key in MODEL_METADATA:
            result = predict_paddy_leaf(
                model_key=model_key,
                image_input=sample_path,
                top_k=3,
                generate_gradcam=True
            )
            self.assertIn("predicted_class", result)
            self.assertIn("confidence", result)
            self.assertEqual(len(result["top_predictions"]), 3)
            self.assertTrue(result["confidence"] > 0)
            self.assertIsNotNone(result["gradcam_superimposed_b64"])
            print(f"  [OK] Model '{model_key}': Predicted '{result['predicted_class']}' ({result['confidence_pct']}) in {result['inference_time_ms']} ms")

    def test_11_api_predict_sample_endpoint(self):
        """Verify POST /api/predict-sample works via AJAX."""
        payload = {
            "filename": "sample_4_brown_spot.jpg",
            "model": "resnet50",
            "gradcam": True
        }
        resp = self.client.post("/api/predict-sample", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["predicted_class"], "brown_spot")

    def test_12_api_benchmark_endpoint(self):
        """Verify POST /api/benchmark runs live latency benchmarking."""
        payload = {
            "model": "efficientnet_b0",
            "warm_up": 5,
            "iterations": 10
        }
        resp = self.client.post("/api/benchmark", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"])
        self.assertTrue(data["data"]["mean_latency_ms"] > 0)
        self.assertTrue(data["data"]["fps"] > 0)
        print(f"  [OK] Live benchmark for EfficientNet-B0: {data['data']['mean_latency_ms']} ms/img ({data['data']['fps']} FPS)")


if __name__ == "__main__":
    unittest.main()
