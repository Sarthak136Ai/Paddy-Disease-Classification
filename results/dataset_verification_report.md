# Dataset Verification & Analysis Report

## 1. Executive Summary
- **Dataset Source**: `anthony2261/paddy-disease-classification`
- **Total Image Samples**: 10,407
- **Corrupted / Unreadable Images**: 0
- **Exact Byte-Duplicate Images**: 74 instances (72 groups)
- **Number of Disease Classes**: 10
- **Color Modes**: {'RGB': 10407}

## 2. Disease Class Distribution

| Class ID | Disease Name | Image Count | Percentage (%) |
|:---:|:---|:---:|:---:|
| 0 | `bacterial_leaf_blight` | 479 | 4.6% |
| 1 | `bacterial_leaf_streak` | 380 | 3.65% |
| 2 | `bacterial_panicle_blight` | 337 | 3.24% |
| 3 | `blast` | 1,738 | 16.7% |
| 4 | `brown_spot` | 965 | 9.27% |
| 5 | `dead_heart` | 1,442 | 13.86% |
| 6 | `downy_mildew` | 620 | 5.96% |
| 7 | `hispa` | 1,594 | 15.32% |
| 8 | `normal` | 1,764 | 16.95% |
| 9 | `tungro` | 1,088 | 10.45% |

## 3. Rice Variety Distribution

| Variety ID | Variety Name | Sample Count | Percentage (%) |
|:---:|:---|:---:|:---:|
| 0 | `ADT45` | 6,992 | 67.19% |
| 1 | `IR20` | 114 | 1.1% |
| 2 | `KarnatakaPonni` | 988 | 9.49% |
| 3 | `Onthanel` | 351 | 3.37% |
| 4 | `Ponni` | 657 | 6.31% |
| 5 | `Surya` | 32 | 0.31% |
| 6 | `Zonal` | 399 | 3.83% |
| 7 | `AndraPonni` | 377 | 3.62% |
| 8 | `AtchayaPonni` | 461 | 4.43% |
| 9 | `RR` | 36 | 0.35% |

## 4. Plant Age (Days) Summary Statistics
- **Min Age**: 45 days
- **Max Age**: 82 days
- **Mean Age**: 64.04 days
- **Median Age**: 67.0 days
- **Standard Deviation**: 8.96 days
- **Unique Age Values**: [45, 47, 50, 55, 57, 60, 62, 65, 66, 67, 68, 70, 72, 73, 75, 77, 80, 82]

## 5. Image Dimension & Resolution Profile
Top image dimensions present in the dataset:
- `480x640`: 10,403 images (99.96%)
- `640x480`: 4 images (0.04%)

## 6. Generated Visualizations
The following plots were generated and saved in `results/plots/dataset_analysis/`:
- `class_distribution.png`: Bar chart of class frequencies and class balance.
- `variety_distribution.png`: Frequency of rice plant varieties.
- `age_distribution.png`: Plant age histogram and KDE.
- `image_dimensions_distribution.png`: Distribution of resolutions and aspect ratios.
- `sample_grid_per_class.png`: 2x5 grid of representative paddy leaf sample images per class.
- `variety_vs_disease_heatmap.png`: Cross-tabulation heatmap of varieties against disease classes.

## 7. Conclusions & Preparation Directives
1. **Data Integrity**: All 10,407 images were successfully loaded without decode corruption.
2. **Class Imbalance**: Normal class has 1,764 samples while bacterial panicle blight has 337 samples. A stratified split is mandatory to ensure equitable representation across train, validation, and test splits.
3. **Resolution Standard**: Images are predominantly 480x640 portrait JPEG images. Resizing to standardized 224x224 RGB tensors with ImageNet normalization is appropriate for deep learning backbones.
4. **Seed & Stratification**: Splitting 70% train (7,284), 15% validation (1,561), and 15% test (1,562) with fixed seed `42` will guarantee reproducibility.