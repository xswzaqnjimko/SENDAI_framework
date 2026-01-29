"""
SENDAI Data Generation using MODIS
===================================

Downloads MODIS satellite imagery for NDVI reconstruction experiments.

MODIS provides DAILY data, giving us 60-90 images per 3-month period.

Datasets used:
- MOD09GA: MODIS Terra Daily Surface Reflectance (500m)
- MYD09GA: MODIS Aqua Daily Surface Reflectance (500m)

Combined: Up to 2 images per day = plenty of data!

Requirements:
- Google Earth Engine account (free): https://earthengine.google.com/
- earthengine-api: pip install earthengine-api

Usage:
    # List available locations
    python data_generation.py --list
    
    # Download data for a specific location
    python data_generation.py --location western_us
    
    # Download data for multiple locations
    python data_generation.py --location western_us australia
    
    # Download all locations (not recommended - do one at a time)
    python data_generation.py
"""

import ee
import numpy as np
import json
import csv
from pathlib import Path
from datetime import datetime
import time
import argparse

# =============================================================================
# CONFIGURATION
# =============================================================================

# Google Earth Engine Project ID - REPLACE WITH YOUR OWN
# Get one for free at: https://console.cloud.google.com/
PROJECT_ID = 'your-gee-project-id'  # <-- CHANGE THIS


# Directory setup
SCRIPT_DIR = Path(__file__).resolve().parent
LOCATIONS_FILE = SCRIPT_DIR / 'locations.csv'

# MODIS configuration
YEAR = 2023
GRID_SIZE = 64      # Output grid size (64x64 pixels)
SCALE = 500         # MODIS resolution in meters
MAX_CLOUD = 30      # Maximum cloud cover percentage
TARGET_IMAGES = 90  # Target number of images per period


# =============================================================================
# LOAD LOCATIONS FROM CSV
# =============================================================================

def load_locations():
    """Load study area locations from locations.csv."""
    locations = {}
    
    if not LOCATIONS_FILE.exists():
        raise FileNotFoundError(
            f"locations.csv not found at {LOCATIONS_FILE}\n"
            "Please ensure locations.csv is in the same directory as this script."
        )
    
    with open(LOCATIONS_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row['location_key']
            locations[key] = {
                'name': row['name'],
                'center': [float(row['center_lon']), float(row['center_lat'])],
                'size_km': int(row['size_km']),
                'sim_months': row['sim_months'],
                'real_months': row['real_months'],
                'model_version': row['model_version'],
            }
    
    return locations


# =============================================================================
# PATH HELPER FUNCTIONS
# =============================================================================

def get_output_dirs(location_key):
    """
    Get output directories for a location.
    
    Output structure:
        data/
        ├── locations.csv
        ├── data_generation.py
        ├── {location_key1}/
        │   ├── processed/
        │   │   ├── simulation_ndvi.npy
        │   │   ├── real_physics_ndvi.npy
        │   │   └── metadata.json
        │   └── visualization/
        │       └── data_preview.png
        └── {location_key2}/...
    """
    processed_dir = SCRIPT_DIR / location_key / 'processed'
    vis_dir = SCRIPT_DIR / location_key / 'visualization'
    
    return {
        'processed': processed_dir,
        'visualization': vis_dir,
    }


# =============================================================================
# INITIALIZE GEE
# =============================================================================

def init_gee():
    """Initialize Google Earth Engine."""
    try:
        ee.Initialize(project=PROJECT_ID)
        print(f"✓ GEE initialized with project: {PROJECT_ID}")
    except Exception as e:
        print(f"Init failed: {e}")
        print("Running authentication...")
        print("\nTo authenticate:")
        print("1. Run: earthengine authenticate")
        print("2. Follow the browser prompts")
        print("3. Re-run this script")
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)
        print("✓ Authenticated and initialized")


# =============================================================================
# MODIS FUNCTIONS
# =============================================================================

def get_study_region(study_area):
    """Create square study region."""
    center = ee.Geometry.Point(study_area["center"])
    buffer_m = study_area["size_km"] * 1000 / 2
    region = center.buffer(buffer_m, maxError=1).bounds()
    return region


def mask_clouds_modis(image):
    """
    Cloud mask for MODIS using state_1km QA band.
    
    Bits 0-1: Cloud state
    00 = clear, 01 = cloudy, 10 = mixed, 11 = not set
    """
    qa = image.select('state_1km')
    cloud_mask = qa.bitwiseAnd(3).eq(0)
    return image.updateMask(cloud_mask)


def add_ndvi_modis(image):
    """
    Add NDVI for MODIS.
    
    MOD09GA bands:
    - sur_refl_b01: Red (620-670nm)
    - sur_refl_b02: NIR (841-876nm)
    
    Scale factor: 0.0001
    """
    red = image.select('sur_refl_b01').multiply(0.0001)
    nir = image.select('sur_refl_b02').multiply(0.0001)
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    ndvi = ndvi.clamp(-1, 1)
    return image.addBands(ndvi)


def get_modis_collection(region, start_date, end_date):
    """Get combined Terra + Aqua MODIS daily surface reflectance."""
    # Terra (morning pass)
    terra = (ee.ImageCollection('MODIS/061/MOD09GA')
             .filterBounds(region)
             .filterDate(start_date, end_date))
    
    # Aqua (afternoon pass)
    aqua = (ee.ImageCollection('MODIS/061/MYD09GA')
            .filterBounds(region)
            .filterDate(start_date, end_date))
    
    # Merge and process
    combined = terra.merge(aqua)
    processed = combined.map(mask_clouds_modis).map(add_ndvi_modis)
    
    return processed.sort('system:time_start')


def get_sample_region(region, grid_size, scale):
    """Get exact region for target grid size."""
    centroid = region.centroid(maxError=1)
    target_size_m = grid_size * scale
    sample_region = centroid.buffer(target_size_m / 2, maxError=1).bounds()
    return sample_region


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_as_numpy(image, region, band='NDVI'):
    """Download a single band as numpy array."""
    import urllib.request
    import tempfile
    import os
    
    url = image.select(band).clip(region).getDownloadURL({
        'scale': SCALE,
        'region': region,
        'format': 'NPY',
        'crs': 'EPSG:4326'
    })
    
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        temp_path = f.name
    
    try:
        urllib.request.urlretrieve(url, temp_path)
        arr = np.load(temp_path, allow_pickle=True)
        
        if hasattr(arr, 'dtype') and arr.dtype.names is not None:
            arr = arr[band].astype(np.float32)
        else:
            arr = arr.astype(np.float32)
        
        return arr
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def download_period_data(region, start_date, end_date, period_name,
                         grid_size=64, target_images=70):
    """Download NDVI data for a time period with equally spaced sampling."""
    print(f"\n{'='*50}")
    print(f"{period_name.upper()}: {start_date} to {end_date}")
    print(f"{'='*50}")
    
    collection = get_modis_collection(region, start_date, end_date)
    n_total = collection.size().getInfo()
    print(f"Total images in collection: {n_total}")
    
    if n_total == 0:
        print("⚠ No images found!")
        return None, []
    
    sample_region = get_sample_region(region, grid_size, SCALE)
    img_list = collection.toList(n_total)
    
    # Calculate equally spaced indices
    oversample_factor = 1.5
    n_candidates = min(int(target_images * oversample_factor), n_total)
    candidate_indices = np.linspace(0, n_total - 1, n_candidates, dtype=int)
    candidate_indices = np.unique(candidate_indices)
    
    print(f"Sampling {len(candidate_indices)} equally-spaced candidates...")
    print(f"Target: {target_images} valid images")
    
    data_list = []
    dates_list = []
    
    for idx_num, i in enumerate(candidate_indices):
        if len(data_list) >= target_images:
            break
        
        try:
            img = ee.Image(img_list.get(int(i)))
            date = img.date().format('YYYY-MM-dd').getInfo()
            
            coverage = img.select('NDVI').reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=sample_region,
                scale=SCALE,
                maxPixels=1e9
            ).get('NDVI').getInfo()
            
            expected_pixels = grid_size * grid_size
            coverage_pct = (coverage or 0) / expected_pixels * 100
            
            if coverage_pct > (100 - MAX_CLOUD):
                arr = download_as_numpy(img, sample_region, 'NDVI')
                
                if arr.shape[0] != grid_size or arr.shape[1] != grid_size:
                    new_arr = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
                    h, w = min(arr.shape[0], grid_size), min(arr.shape[1], grid_size)
                    new_arr[:h, :w] = arr[:h, :w]
                    arr = new_arr
                
                data_list.append(arr)
                dates_list.append(date)
                
                if len(data_list) % 10 == 0:
                    print(f"  Downloaded {len(data_list)}/{target_images}")
            
            time.sleep(0.3)  # Rate limiting
            
        except Exception as e:
            print(f"  [{i}] Error: {str(e)[:60]}")
            continue
    
    if len(data_list) == 0:
        print("✗ No valid images downloaded!")
        return None, []
    
    data = np.stack(data_list, axis=0)
    
    print(f"\n✓ Downloaded {len(data_list)} images")
    print(f"  Shape: {data.shape}")
    print(f"  Date range: {dates_list[0]} to {dates_list[-1]}")
    
    return data, dates_list


def fill_nan_temporal(data):
    """Fill NaN values using temporal interpolation."""
    if data is None:
        return None
    
    data_filled = data.copy()
    T, H, W = data.shape
    
    for i in range(H):
        for j in range(W):
            series = data[:, i, j]
            
            if np.any(np.isnan(series)):
                valid_idx = np.where(~np.isnan(series))[0]
                
                if len(valid_idx) == 0:
                    data_filled[:, i, j] = np.nanmean(data)
                elif len(valid_idx) < len(series):
                    valid_vals = series[valid_idx]
                    all_idx = np.arange(T)
                    interp_vals = np.interp(all_idx, valid_idx, valid_vals)
                    data_filled[:, i, j] = interp_vals
    
    return data_filled


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualization(sim_data, real_data, sim_dates, real_dates, vis_dir, location_name):
    """Create visualization of downloaded data."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        # Row 1: Simulation samples
        indices = [0, len(sim_data)//3, 2*len(sim_data)//3, -1]
        for i, idx in enumerate(indices[:3]):
            ax = axes[0, i]
            ax.imshow(sim_data[idx], cmap='RdYlGn', vmin=-0.1, vmax=0.8)
            ax.set_title(f"Sim: {sim_dates[idx]}", fontsize=10)
            ax.axis('off')
        
        ax = axes[0, 3]
        center = sim_data.shape[1]//2, sim_data.shape[2]//2
        ax.plot(sim_data[:, center[0], center[1]], 'g.-', markersize=2)
        ax.set_title(f'Sim Time Series (n={len(sim_data)})')
        ax.set_ylabel('NDVI')
        ax.set_ylim(-0.1, 0.9)
        
        # Row 2: Real physics samples
        indices = [0, len(real_data)//3, 2*len(real_data)//3, -1]
        for i, idx in enumerate(indices[:3]):
            ax = axes[1, i]
            ax.imshow(real_data[idx], cmap='RdYlGn', vmin=-0.1, vmax=0.8)
            ax.set_title(f"Real: {real_dates[idx]}", fontsize=10)
            ax.axis('off')
        
        ax = axes[1, 3]
        ax.plot(real_data[:, center[0], center[1]], 'b.-', markersize=2)
        ax.set_title(f'Real Time Series (n={len(real_data)})')
        ax.set_ylabel('NDVI')
        ax.set_ylim(-0.1, 0.9)
        
        # Row 3: Statistics
        ax = axes[2, 0]
        ax.imshow(sim_data.mean(axis=0), cmap='RdYlGn', vmin=-0.1, vmax=0.8)
        ax.set_title('Sim Mean NDVI')
        ax.axis('off')
        
        ax = axes[2, 1]
        ax.imshow(real_data.mean(axis=0), cmap='RdYlGn', vmin=-0.1, vmax=0.8)
        ax.set_title('Real Mean NDVI')
        ax.axis('off')
        
        ax = axes[2, 2]
        diff = real_data.mean(axis=0) - sim_data.mean(axis=0)
        im = ax.imshow(diff, cmap='RdBu', vmin=-0.3, vmax=0.3)
        ax.set_title('Difference (Real-Sim)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        ax = axes[2, 3]
        ax.hist(sim_data.flatten(), bins=50, alpha=0.5, label='Sim', density=True)
        ax.hist(real_data.flatten(), bins=50, alpha=0.5, label='Real', density=True)
        ax.set_title('NDVI Distribution')
        ax.legend()
        ax.set_xlabel('NDVI')
        
        plt.suptitle(f'{location_name} - Data Preview', fontsize=14)
        plt.tight_layout()
        plt.savefig(vis_dir / 'data_preview.png', dpi=150)
        plt.close()
        
        print(f"✓ Saved visualization to {vis_dir / 'data_preview.png'}")
        
    except ImportError:
        print("(matplotlib not available, skipping visualization)")


# =============================================================================
# DATE PARSING
# =============================================================================

def get_period_dates(month_range, year):
    """Convert month range string to start/end dates."""
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    
    def days_in_month(month, year):
        if month in [1, 3, 5, 7, 8, 10, 12]:
            return 31
        elif month in [4, 6, 9, 11]:
            return 30
        elif month == 2:
            if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                return 29
            return 28
    
    parts = month_range.split('-')
    start_month = month_map[parts[0]]
    end_month = month_map[parts[1]]
    
    start_date = f"{year}-{start_month:02d}-01"
    end_date = f"{year}-{end_month:02d}-{days_in_month(end_month, year):02d}"
    
    return {'start': start_date, 'end': end_date}


# =============================================================================
# PROCESS SINGLE LOCATION
# =============================================================================

def process_location(location_key, study_area):
    """Process a single location and save data."""
    
    print("\n" + "="*70)
    print(f"Processing: {study_area['name']} ({location_key})")
    print(f"Recommended model: {study_area.get('model_version', 'SENDAI_Jr')}")
    print("="*70)
    
    # Setup directories
    dirs = get_output_dirs(location_key)
    processed_dir = dirs['processed']
    vis_dir = dirs['visualization']
    
    for d in [processed_dir, vis_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput: {processed_dir}")
    
    # Get study region
    region = get_study_region(study_area)
    print(f"\nStudy area: {study_area['name']}")
    print(f"Center: {study_area['center']}")
    print(f"Size: {study_area['size_km']} km")
    
    # Get time periods
    sim_months = study_area['sim_months']
    real_months = study_area['real_months']
    sim_period = get_period_dates(sim_months, YEAR)
    real_period = get_period_dates(real_months, YEAR)
    
    print(f"\nTime periods:")
    print(f"  Simulation: {sim_period['start']} to {sim_period['end']}")
    print(f"  Real physics: {real_period['start']} to {real_period['end']}")
    
    # Download data
    sim_data, sim_dates = download_period_data(
        region, sim_period['start'], sim_period['end'],
        "simulation", GRID_SIZE, TARGET_IMAGES
    )
    
    real_data, real_dates = download_period_data(
        region, real_period['start'], real_period['end'],
        "real_physics", GRID_SIZE, TARGET_IMAGES
    )
    
    if sim_data is None or real_data is None:
        print(f"\n✗ Failed to download sufficient data for {location_key}.")
        return None
    
    # Post-processing
    print("\n--- POST-PROCESSING ---")
    print("Filling NaN values with temporal interpolation...")
    sim_data = fill_nan_temporal(sim_data)
    real_data = fill_nan_temporal(real_data)
    
    print(f"Simulation NaN remaining: {np.isnan(sim_data).sum()}")
    print(f"Real physics NaN remaining: {np.isnan(real_data).sum()}")
    
    # Save
    print("\n--- SAVING ---")
    np.save(processed_dir / 'simulation_ndvi.npy', sim_data.astype(np.float32))
    np.save(processed_dir / 'real_physics_ndvi.npy', real_data.astype(np.float32))
    
    metadata = {
        'study_area': study_area,
        'location_key': location_key,
        'model_version': study_area.get('model_version', 'SENDAI_Jr'),
        'year': YEAR,
        'resolution_m': SCALE,
        'grid_size': GRID_SIZE,
        'simulation': {
            'period': sim_period,
            'months': sim_months,
            'dates': sim_dates,
            'shape': list(sim_data.shape),
            'ndvi_range': [float(np.nanmin(sim_data)), float(np.nanmax(sim_data))],
        },
        'real_physics': {
            'period': real_period,
            'months': real_months,
            'dates': real_dates,
            'shape': list(real_data.shape),
            'ndvi_range': [float(np.nanmin(real_data)), float(np.nanmax(real_data))],
        },
        'created': datetime.now().isoformat(),
    }
    
    with open(processed_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved simulation_ndvi.npy: {sim_data.shape}")
    print(f"✓ Saved real_physics_ndvi.npy: {real_data.shape}")
    print(f"✓ Saved metadata.json")
    
    # Visualization
    print("\n--- VISUALIZATION ---")
    create_visualization(sim_data, real_data, sim_dates, real_dates, 
                        vis_dir, study_area['name'])
    
    # Summary
    print(f"\n{'='*50}")
    print(f"{location_key.upper()} COMPLETE")
    print(f"{'='*50}")
    print(f"""
Output: {processed_dir}

Simulation ({sim_months} {YEAR}):
  Shape: {sim_data.shape}
  Images: {len(sim_dates)}
  NDVI range: [{np.nanmin(sim_data):.3f}, {np.nanmax(sim_data):.3f}]

Real Physics ({real_months} {YEAR}):
  Shape: {real_data.shape}
  Images: {len(real_dates)}
  NDVI range: [{np.nanmin(real_data):.3f}, {np.nanmax(real_data):.3f}]

Next: Run model/{study_area.get('model_version', 'SENDAI_Jr')}.py
""")
    
    return {
        'location_key': location_key,
        'simulation': sim_data,
        'real_physics': real_data,
        'metadata': metadata
    }


# =============================================================================
# MAIN
# =============================================================================

def main(locations=None):
    """Run data generation pipeline for specified locations."""
    
    # Load locations from CSV
    STUDY_AREAS = load_locations()
    
    print("\n" + "="*70)
    print("SENDAI Data Generation (MODIS Daily)")
    print("="*70)
    print(f"Target: {GRID_SIZE}x{GRID_SIZE} grid, {TARGET_IMAGES} images per period")
    
    # Determine which locations to process
    if locations is None:
        locations_to_process = STUDY_AREAS
    elif isinstance(locations, str):
        if locations not in STUDY_AREAS:
            print(f"Error: Unknown location '{locations}'")
            print(f"Available locations: {list(STUDY_AREAS.keys())}")
            return {}
        locations_to_process = {locations: STUDY_AREAS[locations]}
    else:
        locations_to_process = {k: STUDY_AREAS[k] for k in locations if k in STUDY_AREAS}
        unknown = set(locations) - set(STUDY_AREAS.keys())
        if unknown:
            print(f"Warning: Unknown locations ignored: {unknown}")
    
    print(f"\nLocations to process:")
    for key, area in locations_to_process.items():
        print(f"  - {key}: {area['name']}")
        print(f"    Model: {area.get('model_version', 'SENDAI_Jr')}")
    
    # Initialize GEE
    init_gee()
    
    # Process each location
    results = {}
    for location_key, study_area in locations_to_process.items():
        try:
            result = process_location(location_key, study_area)
            if result is not None:
                results[location_key] = result
        except Exception as e:
            print(f"\n✗ Error processing {location_key}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    print("\n" + "="*70)
    print("ALL LOCATIONS COMPLETE")
    print("="*70)
    print(f"\nSuccessfully processed: {len(results)}/{len(locations_to_process)} locations")
    
    for key in results:
        print(f"  ✓ {key}")
    
    failed = set(locations_to_process.keys()) - set(results.keys())
    if failed:
        print(f"\nFailed locations:")
        for key in failed:
            print(f"  ✗ {key}")
    
    print(f"\nNext: Run the appropriate model script on your data")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='SENDAI Data Generation using MODIS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available locations
  python data_generation.py --list
  
  # Process a single location
  python data_generation.py --location western_us
  
  # Process multiple specific locations
  python data_generation.py --location western_us australia
  
  # Process all locations (not recommended)
  python data_generation.py

Output structure:
  data/
  ├── processed/{location}/
  │   ├── simulation_ndvi.npy
  │   ├── real_physics_ndvi.npy
  │   └── metadata.json
  └── visualization/{location}/
      └── data_preview.png
        """
    )
    parser.add_argument(
        '--location', '-l',
        nargs='*',
        help='Specific location(s) to process. If not specified, processes all locations.'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available study areas and exit.'
    )
    
    args = parser.parse_args()
    
    if args.list:
        STUDY_AREAS = load_locations()
        print("\nAvailable study areas:")
        print("-" * 70)
        for key, area in STUDY_AREAS.items():
            print(f"\n{key}:")
            print(f"  Name: {area['name']}")
            print(f"  Center: {area['center']}")
            print(f"  Model: {area.get('model_version', 'SENDAI_Jr')}")
            print(f"  Sim months: {area['sim_months']}")
            print(f"  Real months: {area['real_months']}")
    else:
        results = main(locations=args.location)
