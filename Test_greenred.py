# Theo Kerr on 3/28/2025
# For use on linux cluster "gdalenv" conda env

from All_clean_CHM import *
import argparse
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
parser = argparse.ArgumentParser(description="Script for generating greenred ratio for a CHM")
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
    "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Planet_tiles"]
crs = "EPSG:3857"
pixel_size = 4.77731426716
temp = os.path.join(output_folder, "temp")

print("\n")
print(f"CALCULATING greenred FOR {os.path.basename(input_chm)} ")

# Preprocess CHM
chm_cropped_path, _, _, _, _, planet_cropped_path = preprocess_data_layers(input_chm, temp, data_folders, crs, pixel_size)
    
# Calculate greenred
greenred_path = calc_greenred_by_block(planet_cropped_path, temp, mask=False)
print(f"greenred raster written to {greenred_path}")
print(f"FINISHED CALCULATING greenred FOR {os.path.basename(input_chm)}")
print("\n")