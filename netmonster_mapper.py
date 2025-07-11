import pandas as pd # type: ignore
import numpy as np
from scipy.interpolate import griddata # type: ignore
import folium
from sklearn.neighbors import NearestNeighbors 
import glob
import logging
from typing import List, Optional

def load_and_clean_csv(csv_path):
    """Load NetMonster CSV and clean data."""
    df = pd.read_csv(csv_path)
    
    # Keep essential columns (adjust names based on your CSV)
    required_cols = ['Latitude', 'Longitude', 'RSRP']
    df = df[required_cols].dropna()
    
    # Remove extreme outliers (e.g., RSRP < -120 dBm)
    df = df[(df['RSRP'] >= -120) & (df['RSRP'] <= -50)]
    
    return df

def smooth_gps(df, radius=0.0001):  # ~10 meters
    """Average nearby GPS points to reduce noise."""
    coords = df[['Latitude', 'Longitude']].values
    nbrs = NearestNeighbors(radius=radius).fit(coords)
    smoothed = []
    for i in range(len(coords)):
        neighbors = nbrs.radius_neighbors([coords[i]], return_distance=False)
        smoothed.append(df.iloc[neighbors[0]]['RSRP'].mean())
    df['RSRP'] = smoothed
    return df

def create_heatmap(df, output_kml=None, output_html=None, output_plot=None):
    """Generate a heatmap from RSRP data."""
    lats, lons, rsrp = df['Latitude'], df['Longitude'], df['RSRP']
    
    # Interpolate signal strength onto a grid
    grid_x, grid_y = np.mgrid[
        min(lons):max(lons):100j,
        min(lats):max(lats):100j
    ]
    grid_z = griddata(
        (lons, lats), rsrp,
        (grid_x, grid_y),
        method='cubic', fill_value=-110  # Default for no signal
    )
    
    # --- OPTION 1: Export to KML (Google Earth) ---
    if output_kml:
        import simplekml
        kml = simplekml.Kml()
        for x, y, z in zip(grid_x.ravel(), grid_y.ravel(), grid_z.ravel()):
            pnt = kml.newpoint(coords=[(x, y)])
            pnt.style.iconstyle.color = simplekml.Color.rgb(
                int(255 * (1 - max(0, min(1, (z + 110) / 60)))),  # Red (weak) -> Green (strong)
                int(255 * max(0, min(1, (z + 80) / 60))),
                0
            )
        kml.save(output_kml)
    
    # --- OPTION 2: Interactive Folium Map ---
    if output_html:
        m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=14)
        
        # Add heatmap layer
        from folium.plugins import HeatMap
        heat_data = [[row['Latitude'], row['Longitude'], max(-110, row['RSRP'])] for _, row in df.iterrows()]
        HeatMap(heat_data, radius=15).add_to(m)
        
        # Add markers for weakest/strongest points
        for _, row in df.nlargest(5, 'RSRP').iterrows():  # Strongest
            folium.Marker([row['Latitude'], row['Longitude']], 
                         tooltip=f"Strong: {row['RSRP']} dBm",
                         icon=folium.Icon(color='green')).add_to(m)
        for _, row in df.nsmallest(5, 'RSRP').iterrows():  # Weakest
            folium.Marker([row['Latitude'], row['Longitude']], 
                         tooltip=f"Weak: {row['RSRP']} dBm",
                         icon=folium.Icon(color='red')).add_to(m)
        
        m.save(output_html)

    if output_plot:
        import matplotlib.pyplot as plt
        plt.contourf(grid_x, grid_y, grid_z, levels=20, cmap='RdYlGn_r')
        plt.colorbar(label='RSRP (dBm)')
        plt.scatter(lons, lats, c=rsrp, s=5, edgecolor='black')
        plt.savefig('signal_plot.png')        


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_load_csv(file_path: str) -> Optional[pd.DataFrame]:
    """Attempt to load and clean a CSV with error handling."""
    try:
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_cols = {'Latitude', 'Longitude', 'RSRP'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing columns: {missing}")
            
        # Basic cleaning
        df = df[list(required_cols)].dropna()
        df = df[(df['RSRP'] >= -120) & (df['RSRP'] <= -50)]
        
        return df
    
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {str(e)}")
        return None

def load_all_data(file_pattern: str = "netmonster_*.csv") -> Optional[pd.DataFrame]:
    """Safely load and combine multiple CSVs."""
    try:
        files = glob.glob(file_pattern)
        if not files:
            raise FileNotFoundError(f"No files matching {file_pattern}")
        
        logger.info(f"Found {len(files)} files to process")
        
        # Load with error handling
        all_dfs = []
        for f in files:
            df = safe_load_csv(f)
            if df is not None:
                all_dfs.append(df)
        
        if not all_dfs:
            raise ValueError("No valid data found in any files")
            
        combined_df = pd.concat(all_dfs).drop_duplicates()
        logger.info(f"Combined data shape: {combined_df.shape}")
        
        return combined_df
    
    except Exception as e:
        logger.critical(f"Fatal error during data loading: {str(e)}")
        return None

def main():
    OUTPUT_KML = "signal_map.kml"
    OUTPUT_HTML = "signal_map.html"
    OUTPUT_PLOT = "signal_plot.png"
    
    # Load data with error checking
    combined_df = load_all_data()
    if combined_df is None:
        logger.error("Aborting - no data available")
        return
    
    # Optional smoothing (with check)
    try:
        if len(combined_df) > 10:  # Only smooth if sufficient data
            combined_df = smooth_gps(combined_df)
    except Exception as e:
        logger.warning(f"GPS smoothing failed: {str(e)}")
    
    # Generate outputs
    try:
        create_heatmap(combined_df, OUTPUT_KML, OUTPUT_HTML, OUTPUT_PLOT)
        logger.info(f"Successfully generated map outputs")
    except Exception as e:
        logger.error(f"Failed to generate maps: {str(e)}")

if __name__ == "__main__":
    main()