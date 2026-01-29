# SENDAI: A Hierarchical Sparse-measurement, EfficieNt Data AssImilation Framework

Implementation for the paper "SENDAI: A Hierarchical Sparse-measurement, EfficieNt Data AssImilation Framework".

## Overview

SENDAI is a hierarchical data assimilation framework that reconstructs spatiotemporal fields from hyper-sparse sensor observations by combining simulation-derived priors with learned discrepancy corrections. The framework achieves effective reconstruction from as few as 64 sensors covering 1.56% of the spatial domain.

The architecture decomposes reconstruction into two complementary pathways:
- **Low-frequency pathway**: Leverages Takens' embedding theorem through shallow recurrent decoder networks (SHRED), with latent-space adversarial alignment for domain shift
- **High-frequency pathway**: Employs sequential frequency peeling with coordinate-based implicit neural representations (INRs) to resolve fine-scale structure and sharp boundaries

## Repository Structure

```
SENDAI/
├── README.md
├── data/
│   ├── locations.csv           # Study site coordinates and time windows
│   └── data_generation.py      # MODIS data download via Google Earth Engine
│
├── model/
│   ├── SENDAI_main.py          # Full SENDAI entry point
│   ├── SENDAI_Jr_main.py       # SENDAI Jr. entry point (LF only)
│   └── SENDAI/                 # Core modules
│       ├── __init__.py
│       ├── models.py           # SHRED, DASHRED, HF_SHRED_INR, FullDASHRED
│       ├── training.py         # Training functions
│       ├── data.py             # Data loading, dataset classes
│       ├── metrics.py          # Spatial metrics (SSIM, RMSE, etc.)
│       ├── visualization.py    # Plotting functions
│       └── utils.py            # Training time, model complexity
│
├── demo_videos/                # Reconstruction video demos
│   ├── video_comparison.gif    # Preview image
│   ├── simulation.mp4
│   ├── ground_truth.mp4
│   ├── sendai_jr.mp4
│   └── sendai.mp4
│
└── quick_startup/
    ├── SENDAI_Jr_demo/         # Self-contained demo with pre-downloaded data
    │   ├── SENDAI_Jr_demo.py
    │   ├── SENDAI/             # Module copy
    │   └── data/western_us/
    └── SENDAI_demo/
        ├── SENDAI_demo.py
        ├── SENDAI/             # Module copy
        └── data/northwestern_china/
```

## Getting Started

### Demo Videos

[![SENDAI Reconstruction Demo](demo_videos/video_comparison.gif)]

*To view interactive comparison: Simulation | SENDAI Jr. | SENDAI | Ground Truth*

**Usage**: 
With [interactive video comparison tool](https://github.com/xswzaqnjimko/video-comparison-tool),
Drag/click to upload videos | Drag panels to rearrange
Adjustable playback speed | Enable Overlay Mode for better comparison with click-to-swap or auto-flicker 

### Quick Startup

Pre-downloaded data is provided for immediate testing:

```bash
# SENDAI Jr. (low-frequency pathway only)
cd quick_startup/SENDAI_Jr_demo
python SENDAI_Jr_demo.py

# Full SENDAI (hierarchical multiscale)
cd quick_startup/SENDAI_demo
python SENDAI_demo.py
```

### Full Pipeline

1. **Configure Google Earth Engine** (free academic account)
   ```bash
   pip install earthengine-api
   earthengine authenticate
   ```

2. **Download data**
   ```bash
   cd data
   python data_generation.py --list              # View available sites
   python data_generation.py --location western_us
   ```

3. **Run model**
   ```bash
   cd model
   # Edit LOCATION variable in script, then:
   python SENDAI_Jr_main.py    # Low-frequency pathway
   python SENDAI_main.py       # Full hierarchical model
   ```

## Study Sites

| Location Key | Region | Simulation | Ground Truth | Model |
|--------------|--------|------------|--------------|-------|
| `southwestern_us` | Imperial Valley, CA, USA | Apr–Jun | Jul–Oct | SENDAI |
| `western_us` | Central Valley, CA, USA | Apr–Jun | Jul–Oct | SENDAI Jr. |
| `midwestern_us` | Corn Belt, IA, USA | Apr–Jun | Jul–Oct | SENDAI Jr. |
| `northwestern_china` | Tarim Basin, Xinjiang, China | Apr–Jun | Jul–Oct | SENDAI |
| `western_spain` | Guadalquivir Valley, Spain | Feb–Apr | Sep–Dec | SENDAI Jr. |
| `australia` | Riverina, NSW, Australia | Feb–Apr | Sep–Dec | SENDAI |

## Model Variants

### SENDAI Jr. (Low-Frequency Pathway)

- LSTM encoder with Takens' time-delay embedding
- Latent-space adversarial alignment for sim2real domain shift
- Suitable for landscapes with relatively homogeneous spatial structure

### SENDAI (Full Hierarchical)

- Low-frequency pathway (as above) plus high-frequency correction
- Coordinate-based implicit neural representations with Fourier positional encoding
- Sequential frequency peeling with spectral exclusion
- Recommended for landscapes with sharp boundaries and multi-scale heterogeneity

## Requirements

```
numpy
torch
scikit-learn
scikit-image
scipy
matplotlib
pandas
earthengine-api  # For data download only
```

## Output Files

Model outputs are saved to the `results/` directory:

| File | Description |
|------|-------------|
| `predictions_*.npz` | Reconstructed spatiotemporal fields |
| `comprehensive_metrics.csv` | RMSE, MAE, SSIM, gradient metrics, Moran's I |
| `timing.csv` | Per-stage runtime breakdown |
| `*.pt` | Saved model weights |
| `*.png` | Visualization figures |

## Citation

```
Paper under review.
```

## Data Sources

- MODIS Terra/Aqua Daily Surface Reflectance (MOD09GA/MYD09GA)
- Google Earth Engine for data access and preprocessing
