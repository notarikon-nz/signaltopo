# SignalTopo

**SignalTopo** is a Python tool for **visualizing and analyzing cellular signal strength data** collected via NetMonster exports.

It generates:
- **KML files** for Google Earth visualization  
- **Interactive HTML heatmaps** with Folium  
- **Contour plots** for signal strength analysis

## Features

- Loads and validates NetMonster CSV exports automatically.
- Smooths GPS and RSRP data for cleaner visualizations.
- Generates heatmaps, KML overlays, and contour plots with a single run.
- Optimized for **speed and low memory usage** on large datasets.

## Installation

```bash
git clone https://github.com/notarikon-nz/signaltopo.git
cd signaltopo
pip install -r requirements.txt
```