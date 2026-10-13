"""
Quick Verification Script for Running Flask Application.
"""

import urllib.request
import json

endpoints = [
    "/",
    "/predict",
    "/model-performance",
    "/class-performance",
    "/class-performance?model=efficientnet_b0",
    "/confusion-matrix",
    "/confusion-matrix?model=resnet50",
    "/model-selection",
    "/about",
    "/api/samples",
    "/api/metrics",
    "/api/metrics/resnet50"
]

print("=" * 70)
print("  VERIFYING FLASK SERVER ENDPOINTS (http://127.0.0.1:5000)")
print("=" * 70)

for ep in endpoints:
    url = f"http://127.0.0.1:5000{ep}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            status = resp.status
            content_type = resp.headers.get("Content-Type", "").split(";")[0]
            data = resp.read()
            print(f"  [OK {status}] {ep:<42} ({len(data):>6} bytes, {content_type})")
    except Exception as e:
        print(f"  [FAIL] {ep:<42} Error: {e}")

# Test Live Inference with Grad-CAM via POST /api/predict-sample
try:
    sample_url = "http://127.0.0.1:5000/api/predict-sample"
    payload = json.dumps({
        "filename": "sample_3_blast.jpg",
        "model": "resnet50",
        "gradcam": True
    }).encode("utf-8")
    req = urllib.request.Request(sample_url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        d = res["data"]
        print("\n" + "-" * 70)
        print("  LIVE INFERENCE TEST (ResNet50 + Grad-CAM):")
        print(f"    Target Image:    sample_3_blast.jpg")
        print(f"    Predicted Class: {d['display_name']} ({d['predicted_class']})")
        print(f"    Confidence:      {d['confidence_pct']}")
        print(f"    Latency:         {d['inference_time_ms']} ms on {d['device']}")
        print(f"    Top-3 Breakdown: {[x['display_name'] + ' (' + x['confidence_pct'] + ')' for x in d['top_predictions']]}")
        print(f"    Grad-CAM Heatmap:{'Generated (Data URL Present)' if d.get('gradcam_superimposed_b64') else 'None'}")
        print(f"    Diagnosis Info:  {d['disease_info']['scientific_name']}")
except Exception as e:
    print(f"  [FAIL API] POST /api/predict-sample Error: {e}")

# Test Live Benchmarking via POST /api/benchmark
try:
    bench_url = "http://127.0.0.1:5000/api/benchmark"
    payload = json.dumps({
        "model": "efficientnet_b0",
        "warm_up": 10,
        "iterations": 30
    }).encode("utf-8")
    req = urllib.request.Request(bench_url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        d = res["data"]
        print("\n" + "-" * 70)
        print("  LIVE BENCHMARK TEST (EfficientNet-B0):")
        print(f"    Device:          {d['device_name']} ({d['device']})")
        print(f"    Mean Latency:    {d['mean_latency_ms']} ms/image")
        print(f"    Median Latency:  {d['median_latency_ms']} ms/image")
        print(f"    Std Deviation:   ±{d['std_latency_ms']} ms")
        print(f"    Throughput:      {d['fps']} images/second")
        print(f"    Recorded Ref:    {d['recorded_latency_ms']} ms/image")
except Exception as e:
    print(f"  [FAIL API] POST /api/benchmark Error: {e}")

print("=" * 70 + "\n")
