import glob
import logging
from typing import List, Optional

import folium
import numpy as np
import pandas as pd
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dataframe contains required columns and valid RSRP values."""
    missing_cols = REQUIRED_COLS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    filtered_df = df[list(REQUIRED_COLS)].dropna()
    return filtered_df[
        (filtered_df['RSRP'] >= RSRP_RANGE[0]) & 
        (filtered_df['RSRP'] <= RSRP_RANGE[1])
    ]


def load_signal_data(file_path: str) -> Optional[pd.DataFrame]:
    """Load and validate signal data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        return validate_dataframe(df)
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {str(e)}")
        return None


def smooth_gps_coordinates(df: pd.DataFrame, radius: float = GPS_SMOOTHING_RADIUS) -> pd.DataFrame:
    """Apply radius-based smoothing to GPS coordinates."""
    coords = df[['Latitude', 'Longitude']].values
    nbrs = NearestNeighbors(radius=radius).fit(coords)
    
    smoothed_rsrp = [
        df.iloc[nbrs.radius_neighbors([coord], return_distance=False)[0]]['RSRP'].mean()
        for coord in coords
    ]
    
    df['RSRP'] = smoothed_rsrp
    return df


def create_interpolation_grid(df: pd.DataFrame) -> tuple:
    """Create grid coordinates for signal interpolation."""
    lats, lons = df['Latitude'], df['Longitude']
    return np.mgrid[
        min(lons):max(lons):GRID_RESOLUTION,
        min(lats):max(lats):GRID_RESOLUTION
    ]


def generate_kml_output(grid_points: np.ndarray, signal_strengths: np.ndarray, output_path: str) -> None:
    """Generate KML file for Google Earth visualization."""
    import simplekml
    kml = simplekml.Kml()
    
    for (x, y), z in zip(grid_points, signal_strengths):
        point = kml.newpoint(coords=[(x, y)])
        normalized_strength = max(0, min(1, (z + 110) / 60))
        point.style.iconstyle.color = simplekml.Color.rgb(
            int(255 * (1 - normalized_strength)),
            int(255 * max(0, min(1, (z + 80) / 60))),
            0
        )
    
    kml.save(output_path)


def generate_folium_map(df: pd.DataFrame, output_path: str) -> None:
    """Generate interactive Folium heatmap."""
    map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
    m = folium.Map(location=map_center, zoom_start=14)
    
    heat_data = [
        [row['Latitude'], row['Longitude'], max(NO_SIGNAL_DEFAULT, row['RSRP'])] 
        for _, row in df.iterrows()
    ]
    HeatMap(heat_data, radius=HEATMAP_RADIUS).add_to(m)
    
    for strength, color, count in [('Strong', 'green', 5), ('Weak', 'red', 5)]:
        data_subset = df.nlargest(count, 'RSRP') if strength == 'Strong' else df.nsmallest(count, 'RSRP')
        for _, row in data_subset.iterrows():
            folium.Marker(
                [row['Latitude'], row['Longitude']],
                tooltip=f"{strength}: {row['RSRP']} dBm",
                icon=folium.Icon(color=color)
            ).add_to(m)
    
    m.save(output_path)


def generate_contour_plot(grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray, output_path: str) -> None:
    """Generate matplotlib contour plot."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    plt.contourf(grid_x, grid_y, grid_z, levels=20, cmap='RdYlGn_r')
    plt.colorbar(label='RSRP (dBm)')
    plt.savefig(output_path)
    plt.close()


def process_signal_data(df: pd.DataFrame) -> tuple:
    """Process signal data and create interpolation grid."""
    grid_x, grid_y = create_interpolation_grid(df)
    grid_z = griddata(
        (df['Longitude'], df['Latitude']), df['RSRP'],
        (grid_x, grid_y),
        method='cubic', fill_value=NO_SIGNAL_DEFAULT
    )
    return grid_x, grid_y, grid_z


def load_multiple_files(file_pattern: str = "netmonster_*.csv") -> Optional[pd.DataFrame]:
    """Load and combine multiple signal data files."""
    try:
        files = glob.glob(file_pattern)
        if not files:
            raise FileNotFoundError(f"No files matching {file_pattern}")
        
        logger.info(f"Found {len(files)} files to process")
        data_frames = [df for f in files if (df := load_signal_data(f)) is not None]
        
        if not data_frames:
            raise ValueError("No valid data found in any files")
            
        combined_df = pd.concat(data_frames).drop_duplicates()
        logger.info(f"Combined data shape: {combined_df.shape}")
        return combined_df
    
    except Exception as e:
        logger.critical(f"Data loading failed: {str(e)}")
        return None


def main() -> None:
    """Main execution function."""
    output_files = {
        'kml': "signal_map.kml",
        'html': "signal_map.html",
        'plot': "signal_plot.png"
    }
    
    signal_data = load_multiple_files()
    if signal_data is None:
        logger.error("Aborting - no valid data available")
        return
    
    try:
        if len(signal_data) > 10:
            signal_data = smooth_gps_coordinates(signal_data)
    except Exception as e:
        logger.warning(f"GPS smoothing skipped: {str(e)}")
    
    try:
        grid_x, grid_y, grid_z = process_signal_data(signal_data)
        
        if output_files['kml']:
            generate_kml_output(
                np.column_stack((grid_x.ravel(), grid_y.ravel())), 
                grid_z.ravel(), 
                output_files['kml']
            )
        
        if output_files['html']:
            generate_folium_map(signal_data, output_files['html'])
        
        if output_files['plot']:
            generate_contour_plot(grid_x, grid_y, grid_z, output_files['plot'])
        
        logger.info("Successfully generated all output files")
    except Exception as e:
        logger.error(f"Map generation failed: {str(e)}")


if __name__ == "__main__":
    main()