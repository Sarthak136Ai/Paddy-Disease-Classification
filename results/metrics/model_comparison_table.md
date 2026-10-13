# Model Performance Benchmark on Untouched Test Set (N=1,562)

| Model | Test Accuracy (%) | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Parameters | Model Size (MB) | Inference Latency (ms/img) | Test Loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet50 | 96.93 | 0.9694 | 0.97 | 0.9696 | 0.9692 | 23528522 | 269.71 | 9.44 | 0.1428 |
| EfficientNet-B0 | 96.73 | 0.9642 | 0.9668 | 0.9654 | 0.9674 | 4020358 | 46.48 | 7.08 | 0.1463 |
| MobileNetV3 | 94.37 | 0.9333 | 0.942 | 0.9371 | 0.944 | 4214842 | 48.58 | 11.63 | 0.2128 |
| DenseNet121 | 92.57 | 0.9207 | 0.9203 | 0.9199 | 0.9257 | 6964106 | 80.59 | 9.85 | 0.2775 |
| Custom CNN (Baseline) | 45.39 | 0.3395 | 0.3332 | 0.3163 | 0.415 | 423338 | 4.88 | 10.82 | 1.6005 |

## Key Architectural Insights
1. **Top Classifier**: `ResNet50` achieved the highest Macro F1 (**0.9696**) and Test Accuracy (**96.93%**).
2. **Best Efficiency & Deployment Choice**: `EfficientNet-B0` achieved near-identical accuracy (**96.73%** / **0.9654** Macro F1) with **5.8x fewer parameters** (4.02M vs 23.53M) and the fastest inference latency (**7.08 ms/image**).
3. **Baseline Scratch Gap**: The scratch `Custom CNN` baseline achieved **45.39%** accuracy and **0.3163** Macro F1, demonstrating the crucial importance of ImageNet transfer learning priors for recognizing nuanced foliar leaf pathogens.
