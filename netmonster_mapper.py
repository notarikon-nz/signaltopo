import glob
from pathlib import Path
import logging
from typing import List, Optional, Tuple
from functools import lru_cache
import warnings
import time

import folium
import numpy as np
import pandas as pd
from contextlib import contextmanager
from scipy import sparse
from folium.plugins import HeatMap
from scipy.interpolate import griddata
from sklearn.neighbors import NearestNeighbors

# Constants
REQUIRED_COLS = {'Latitude', 'Longitude', 'RSRP'}
RSRP_RANGE = (-120, -50)
GPS_SMOOTHING_RADIUS = 0.0001  # ~10 meters
GRID_RESOLUTION = 100j
HEATMAP_RADIUS = 15
NO_SIGNAL_DEFAULT = -110
SMOOTHING_THRESHOLD = 10
CHUNK_SIZE = 10000  # Process data in chunks to manage memory
MAX_INTERPOLATION_POINTS = 50000  # Limit interpolation points for performance

# Configure logging
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

@contextmanager
def timed_operation(description: str):
    start = time.time()
    yield
    logger.info(f"{description} completed in {time.time()-start:.2f}s")


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate dataframe with optimized filtering and type conversion.
    
    Optimizations:
    - Early column validation to fail fast
    - Vectorized operations for RSRP filtering
    - Efficient data type optimization
    """
    missing_cols = REQUIRED_COLS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Select only required columns early to reduce memory footprint
    filtered_df = df[list(REQUIRED_COLS)].copy()
    
    # Drop NaN values efficiently
    filtered_df.dropna(inplace=True)
    
    # Vectorized RSRP range filtering
    rsrp_mask = (filtered_df['RSRP'] >= RSRP_RANGE[0]) & (filtered_df['RSRP'] <= RSRP_RANGE[1])
    filtered_df = filtered_df[rsrp_mask]
    
    # Optimize data types to reduce memory usage
    filtered_df['RSRP'] = filtered_df['RSRP'].astype(np.float32)
    filtered_df['Latitude'] = filtered_df['Latitude'].astype(np.float64)
    filtered_df['Longitude'] = filtered_df['Longitude'].astype(np.float64)
    
    return filtered_df


def load_signal_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load and validate signal data with optimized CSV reading.
    
    Optimizations:
    - Specify dtypes during CSV reading for faster parsing
    - Use only required columns to reduce memory usage
    """
    try:
        # Pre-specify dtypes for faster CSV parsing
        dtype_dict = {
            'Latitude': np.float64,
            'Longitude': np.float64,
            'RSRP': np.float32
        }
        
        # Load only required columns if they exist
        df = pd.read_csv(
            file_path,
            usecols=lambda x: x in REQUIRED_COLS,
            dtype=dtype_dict,
            engine='c'  # Use C engine for better performance
        )
        
        return validate_dataframe(df)
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {str(e)}")
        return None


def smooth_gps_coordinates(df: pd.DataFrame, radius: float = GPS_SMOOTHING_RADIUS) -> pd.DataFrame:
    """
    Optimized GPS coordinate smoothing with memory-efficient operations.
    
    Optimizations:
    - Pre-allocate arrays to avoid repeated memory allocation
    - Use sparse matrices efficiently
    - Vectorized operations where possible
    - Early termination for small datasets
    """
    if len(df) < SMOOTHING_THRESHOLD:
        return df
    
    # Use float32 for coordinates to reduce memory usage
    coords = df[['Latitude', 'Longitude']].values.astype(np.float32)
    
    # Configure NearestNeighbors for optimal performance
    nbrs = NearestNeighbors(
        radius=radius, 
        algorithm='ball_tree',  # More efficient for geographic data
        leaf_size=30,  # Optimized leaf size
        n_jobs=-1
    ).fit(coords)
    
    # Get sparse adjacency matrix
    adjacency = nbrs.radius_neighbors_graph(coords, mode='connectivity')
    rsrp_array = df['RSRP'].values.astype(np.float32)
    
    # Vectorized smoothing calculation
    sums = adjacency @ rsrp_array
    counts = adjacency @ np.ones_like(rsrp_array)
    
    # Avoid division by zero with efficient masking
    valid_mask = counts != 0
    smoothed = rsrp_array.copy()
    smoothed[valid_mask] = sums[valid_mask] / counts[valid_mask]
    
    # Return modified copy efficiently
    result_df = df.copy()
    result_df['RSRP'] = smoothed
    return result_df


@lru_cache(maxsize=32)
def create_interpolation_grid_cached(lat_min: float, lat_max: float, 
                                   lon_min: float, lon_max: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cached grid creation to avoid recomputation for similar bounds.
    
    Optimizations:
    - LRU cache for repeated grid generation
    - Separate function for cacheable operations
    """
    return np.mgrid[lon_min:lon_max:GRID_RESOLUTION, lat_min:lat_max:GRID_RESOLUTION]


def create_interpolation_grid(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create grid coordinates for signal interpolation with caching.
    
    Optimizations:
    - Use cached grid generation
    - Efficient min/max calculation
    """
    # Calculate bounds efficiently
    lat_bounds = df['Latitude'].agg(['min', 'max'])
    lon_bounds = df['Longitude'].agg(['min', 'max'])
    
    return create_interpolation_grid_cached(
        lat_bounds['min'], lat_bounds['max'],
        lon_bounds['min'], lon_bounds['max']
    )


def generate_kml_output(grid_points: np.ndarray, signal_strengths: np.ndarray, 
                                output_path: str) -> None:
    """
    Optimized KML generation with vectorized color calculations.
    
    Optimizations:
    - Vectorized color computation
    - Efficient array operations
    - Lazy import for better startup time
    """
    try:
        import simplekml
    except ImportError:
        logger.warning("simplekml not available, skipping KML generation")
        return
    
    kml = simplekml.Kml()
    
    # Vectorized color calculation
    normalized_strengths = np.clip((signal_strengths + 110) / 60, 0, 1)
    red_values = (255 * (1 - normalized_strengths)).astype(np.uint8)
    green_values = (255 * np.clip((signal_strengths + 80) / 60, 0, 1)).astype(np.uint8)
    
    # Batch point creation (more efficient than individual operations)
    for i, (x, y) in enumerate(grid_points):
        point = kml.newpoint(coords=[(float(x), float(y))])
        point.style.iconstyle.color = simplekml.Color.rgb(
            int(red_values[i]), int(green_values[i]), 0
        )
    
    kml.save(output_path)


def generate_folium_map(df: pd.DataFrame, output_path: str) -> None:
    """
    Optimized Folium heatmap generation with efficient data preparation.
    
    Optimizations:
    - Vectorized heat data preparation
    - Efficient sampling for markers
    - Reduced memory allocations
    """
    # Calculate center efficiently
    map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
    m = folium.Map(location=map_center, zoom_start=14)
    
    # Vectorized heat data preparation
    heat_data = np.column_stack([
        df['Latitude'].values,
        df['Longitude'].values,
        np.maximum(NO_SIGNAL_DEFAULT, df['RSRP'].values)
    ]).tolist()
    
    HeatMap(heat_data, radius=HEATMAP_RADIUS).add_to(m)
    
    # Efficient marker placement using nlargest/nsmallest
    for strength, color, is_strong in [('Strong', 'green', True), ('Weak', 'red', False)]:
        data_subset = df.nlargest(5, 'RSRP') if is_strong else df.nsmallest(5, 'RSRP')
        
        for _, row in data_subset.iterrows():
            folium.Marker(
                [row['Latitude'], row['Longitude']],
                tooltip=f"{strength}: {row['RSRP']:.1f} dBm",
                icon=folium.Icon(color=color)
            ).add_to(m)
    
    m.save(output_path)


def generate_contour_plot(grid_x: np.ndarray, grid_y: np.ndarray, 
                                  grid_z: np.ndarray, output_path: str) -> None:
    """
    Optimized contour plot generation with lazy import and efficient plotting.
    
    Optimizations:
    - Lazy matplotlib import
    - Efficient figure handling
    - Memory-conscious plotting
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for better performance
    except ImportError:
        logger.warning("matplotlib not available, skipping contour plot generation")
        return
    
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(grid_x, grid_y, grid_z, levels=20, cmap='RdYlGn_r')
    plt.colorbar(contour, label='RSRP (dBm)')
    plt.title('Signal Strength Contour Map')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_signal_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimized signal data processing with adaptive sampling.
    
    Optimizations:
    - Adaptive data sampling for large datasets
    - Efficient interpolation method selection
    - Memory-conscious grid generation
    """
    # Sample data if too large for efficient interpolation
    if len(df) > MAX_INTERPOLATION_POINTS:
        logger.info(f"Sampling {MAX_INTERPOLATION_POINTS} points from {len(df)} for interpolation")
        df = df.sample(n=MAX_INTERPOLATION_POINTS, random_state=42)
    
    grid_x, grid_y = create_interpolation_grid(df)
    
    # Choose interpolation method based on data size
    method = 'linear' if len(df) > 10000 else 'cubic'
    
    # Suppress scipy warnings for cleaner output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid_z = griddata(
            (df['Longitude'].values, df['Latitude'].values), 
            df['RSRP'].values,
            (grid_x, grid_y),
            method=method, 
            fill_value=NO_SIGNAL_DEFAULT
        )
    
    return grid_x, grid_y, grid_z


def load_multiple_files(file_pattern: str = "netmonster_*.csv") -> Optional[pd.DataFrame]:
    """
    Optimized multi-file loading with memory-efficient concatenation.
    
    Optimizations:
    - Process files in chunks to manage memory
    - Efficient deduplication strategy
    - Early validation and filtering
    """
    try:
        files = list(Path().glob(file_pattern))
        if not files:
            raise FileNotFoundError(f"No files matching {file_pattern}")
        
        logger.info(f"Found {len(files)} files to process")
        
        # Process files in chunks to manage memory
        all_data = []
        for file_path in files:
            df = load_signal_data(file_path)
            if df is not None and len(df) > 0:
                all_data.append(df)
        
        if not all_data:
            raise ValueError("No valid data found in any files")
        
        # Efficient concatenation with ignore_index for better performance
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Efficient deduplication using subset of columns
        initial_size = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['Latitude', 'Longitude'], keep='first')
        
        logger.info(f"Combined data: {initial_size} -> {len(combined_df)} points after deduplication")
        return combined_df
    
    except Exception as e:
        logger.critical(f"Data loading failed: {str(e)}")
        return None


def main() -> None:
    """
    Main execution function with optimized workflow.
    
    Optimizations:
    - Conditional processing based on data size
    - Parallel-safe operations
    - Efficient resource management
    """
    output_files = {
        'kml': "signal_map.kml",
        'html': "signal_map.html",
        'plot': "signal_plot.png"
    }
    
    with timed_operation("Total execution time"):

        # Load data with optimized loading
        signal_data = load_multiple_files()
        if signal_data is None:
            logger.error("Aborting - no valid data available")
            return
        
        logger.info(f"Processing {len(signal_data)} data points")
        
        # Apply GPS smoothing only for datasets that benefit from it
        try:
            if len(signal_data) > SMOOTHING_THRESHOLD:
                with timed_operation("Smoothing GPS coordinates"):
                    signal_data = smooth_gps_coordinates(signal_data)

        except Exception as e:
            logger.warning(f"GPS smoothing skipped: {str(e)}")
        
        # Generate outputs with optimized functions
        try:
            with timed_operation("Interpolation Grid generation"):
                grid_x, grid_y, grid_z = process_signal_data(signal_data)
            
            if output_files['kml']:
                with timed_operation("KML generation"):
                    generate_kml_output(
                        np.column_stack((grid_x.ravel(), grid_y.ravel())), 
                        grid_z.ravel(), 
                        output_files['kml']
                    )
            
            if output_files['html']:
                with timed_operation("HTML heatmap generation"):
                    generate_folium_map(signal_data, output_files['html'])
            
            if output_files['plot']:
                with timed_operation("Contour plot generation"):
                    generate_contour_plot(grid_x, grid_y, grid_z, output_files['plot'])
            
            logger.info("Successfully generated all output files")
        except Exception as e:
            logger.error(f"Map generation failed: {str(e)}")


if __name__ == "__main__":
    main()