# Paddy Leaf Disease Dataset Specification

## Overview
This dataset contains paddy/rice leaf images categorized across 10 distinct disease and healthy states. It originates from the Hugging Face hub repository `anthony2261/paddy-disease-classification`.

## Dataset Attributes
- **Total Images**: 10,407
- **Number of Classes**: 10
- **Image Modality**: RGB Photography
- **Default Resolution**: 480 × 640 pixels (Portrait)

## Class Inventory
0. `bacterial_leaf_blight`
1. `bacterial_leaf_streak`
2. `bacterial_panicle_blight`
3. `blast`
4. `brown_spot`
5. `dead_heart`
6. `downy_mildew`
7. `hispa`
8. `normal`
9. `tungro`

## Metadata
- `variety`: Rice cultivars (ADT45, IR20, KarnatakaPonni, Onthanel, Ponni, Surya, Zonal, AndraPonni, AtchayaPonni, RR)
- `age`: Plant age recorded in days

## Storage Layout
- `data/splits/`: Stratified split index mappings (`split_indices.json`) guaranteeing strict 70/15/15 train/val/test partitions.
