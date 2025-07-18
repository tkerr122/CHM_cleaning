# Theo Kerr on 3/28/2025
# For use on linux cluster "gdalenv" conda env

from All_clean_CHM import *
import argparse
import pandas as pd
gdal.UseExceptions()


"""I have written this script to be a command-line utility for creating an NDVI raster cropped to the 
extent of the CHM raster. This should be the first step when looking to clean a dataset of slope errors,
since it allows the user to find what the appropriate height and NDVI thresholds should be for the 
filtering.
================================================
-s option: survey name

Assumes the following input variables are hardcoded:
 - input_chm
 - data_folders
 - output_folder
 - crs
 - pixel_size

"""
# Create argument parser
parser = argparse.ArgumentParser(description="Script for generating redgreen ratio for a CHM")
parser.add_argument("-s", "--survey", type=str, help="Survey name", required=True)

# Parse arguments
args = parser.parse_args()

# Set up variables
survey = args.survey
input_chm = f"/gpfs/glad1/Theo/Data/Lidar/CHMs_raw/1_Combined_CHMs/{survey}_CHM.tif"
output_folder = f"/gpfs/glad1/Theo/Data/Lidar/CHM_testing/{survey}"
data_folders = ["/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Canopy/Canopy.shp", 
    "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Powerlines", 
    "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Manual_powerlines", 
    "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Water", 
    "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Landsat", 
    "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Slope_errors/Slope_errors.shp",
    "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/WorldCover",
    "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Planet_tile_list/Planet_tile_list.csv"]
crs = "EPSG:3857"
pixel_size = 4.77731426716
temp = os.path.join(output_folder, "temp")

print("\n")
print(f"CALCULATING REDGREEN FOR {os.path.basename(input_chm)} ")

# Test get_chm_loc
def get_chm_loc2(chm):
    """Takes given raster dataset and finds which lat lon tiles it intersects with, returns a list of the tiles.

    Args:
        chm (GDAL raster): CHM raster.

    Returns:
        list: list of the tiles intersecting with the CHM.
    """
    # Get CHM geotransform info
    gt = chm.GetGeoTransform()
    xsize = chm.RasterXSize
    ysize = chm.RasterYSize
    
    x_left = gt[0]
    y_top = gt[3]
    
    x_right = x_left + (xsize * gt[1])
    y_bottom = y_top + (ysize * gt[5])
    
    # Find what lat lon tile it intersects with
    src_srs = osr.SpatialReference()
    src_srs.ImportFromEPSG(3857)
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(4326)
    transform = osr.CoordinateTransformation(src_srs, dst_srs)
    lat_min, lon_min, _ = transform.TransformPoint(x_left, y_bottom)
    lat_max, lon_max, _ = transform.TransformPoint(x_right, y_top)
    
    # Extract tile names
    def get_tile_name(lat_min, lat_max, lon_min, lon_max):
        tile_names = []
        
        lat_start = math.ceil(lat_max / 10) * 10
        lat_end = math.ceil(lat_min / 10) * 10
        
        lon_start = math.floor(lon_min / 10) * 10
        lon_end = math.floor(lon_max / 10) * 10
        
        for lat in range(lat_start, lat_end - 1, -10):
            for lon in range(lon_start, lon_end + 1, 10):
                lat_dir = "N" if lat >= 0 else "S"
                lon_dir = "E" if lon >= 0 else "W"
        
                lat_str = f"{abs(lat):02d}{lat_dir}"
                lon_str = f"{abs(lon):03d}{lon_dir}"
                
                tile_name = f"{lat_str}_{lon_str}"
                tile_names.append(tile_name)
        
        return tile_names
    
    # Extract WorldCover tile names
    def get_wc_tile_name(lat_min, lat_max, lon_min, lon_max):
        wc_names = []
        
        lat_start = math.floor(lat_min / 3) * 3
        lat_end = math.floor(lat_max / 3) * 3
        
        lon_start = math.floor(lon_min / 3) * 3
        lon_end = math.floor(lon_max / 3) * 3
        
        for lat in range(lat_start, lat_end + 1, 3):
            for lon in range(lon_start, lon_end + 1, 3):
                lat_dir = "N" if lat > 0 else "S"
                lon_dir = "E" if lon > 0 else "W"
        
                lat_str = f"{lat_dir}{abs(lat):02d}"
                lon_str = f"{lon_dir}{abs(lon):03d}"
                
                tile_name = f"ESA_WorldCover_10m_2021_v200_{lat_str}{lon_str}_Map.tif"
                wc_names.append(tile_name)
        
        return wc_names
    
    # Extract planet tiles
    def get_planet_tile_name(tiles):
        planet_tiles = pd.read_csv(data_folders[7])
        planet_tiles = planet_tiles[planet_tiles['tile_name'].isin(tiles)]
        
        # test csv
        planet_tiles['location'].to_csv("/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Planet_tile_list/test.csv", index=False)
        
        planet_tile_names = planet_tiles['location'].tolist()
        
        return planet_tile_names
        
    tiles = get_tile_name(lat_min, lat_max, lon_min, lon_max)
    tiles = sorted(set(tiles))
    
    wc_tiles = get_wc_tile_name(lat_min, lat_max, lon_min, lon_max)
    wc_tiles = sorted(set(wc_tiles))
    
    planet_tiles = get_planet_tile_name(tiles)
    planet_tiles = sorted(set(planet_tiles))
        
    return tiles, wc_tiles, planet_tiles

chm = gdal.Open(input_chm)
tiles, wc_tiles, planet_tiles = get_chm_loc2(chm)
print(f"Tiles: {tiles}")
print(f"WC tiles: {tiles}")
# print(f"Planet tiles: {planet_tiles}")

# # Preprocess CHM
# chm_cropped_path, powerlines_cropped_path, water_cropped_path, landsat_cropped_path, _, _, _ = preprocess_data_layers(input_chm, temp, data_folders, crs, pixel_size)
    
# # Calculate NDVI
# redgreen_path = calc_redgreen_by_block(landsat_cropped_path, temp, mask=False)
# print(f"Redgreen raster written to {redgreen_path}")
print(f"FINISHED CALCULATING REDGREEN FOR {os.path.basename(input_chm)}")
print("\n")