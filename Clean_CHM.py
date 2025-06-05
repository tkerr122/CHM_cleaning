# Theo Kerr on 3/7/2025
# For use on linux cluster "gdalenv" conda env

# Imports/env settings 
from All_clean_CHM import *
import argparse
gdal.UseExceptions()

"""I have written this script to be a command-line utility for cleaning a CHM of powerlines, water, and if
desired, slope errors, using specific values for height and NDVI thresholds. 
================================================
-s option: survey name.
-bs option: desired buffer size for powerlines, in meters. Defaults to 50.
-st option: whether or not to save the temp rasters.
-sm option: whether or not to mask out the slope errors.
-mp option: whether or not to use a manual powerline file for additional powerline masking.
-ms option: whether or not to use a manual slope errors shapefile for slope masking.
-ht option: height threshold (values above this number are considered) used for slope masking. Defaults to 120.
-nt option: NDVI threshold (values above this are kept). Defaults to 120.
-wmv option: list of pixel values from water image to use when masking water. 

Assumes the following input variables are hardcoded:
 - input_chm
 - data_folders
 - output_folder
 - crs
 - pixel_size

"""

# Create argument parser
parser = argparse.ArgumentParser(description="Script for cleaning single CHM")
parser.add_argument("-s", "--survey", type=str, help="Survey name", required=True)
parser.add_argument("-bs", "--buffer-size", type=int, default=50, help="Buffer size")
parser.add_argument("-st", "--save-temp", action="store_true", help="Save temp dir")
parser.add_argument("-sm", "--slope-mask", action="store_true", help="Filter out slope")
parser.add_argument("-mp", "--man-pwl", action="store_true", help="Buffer manual powerlines")
parser.add_argument("-ms", "--man-slp", action="store_true", help="Mask with manual slope")
parser.add_argument("-ht", "--height-threshold", type=int, help="Height threshold")
parser.add_argument("-nt", "--ndvi-threshold", type=int, default=120, help="NDVI threshold")
parser.add_argument("-wmv", "--water-mask-values", type=list, default=[2, 3, 4, 5, 6, 7, 8, 11], help="Water mask values")
parser.add_argument("-wcmv", "--wc-mask-values", type=list, default=[60, 70, 80], help="WorldCover mask values")
parser.add_argument("-wcndv", "--wc-nodata-values", type=list, default=[50], help="WorldCover nodata values")

# Parse arguments
args = parser.parse_args()

# Set up variables
survey = args.survey
buffer_size = args.buffer_size
save_temp = args.save_temp
slope_mask = args.slope_mask
man_pwl = args.man_pwl
man_slp = args.man_slp
height_threshold = args.height_threshold
ndvi_threshold = args.ndvi_threshold
water_mask_values = args.water_mask_values
wc_mask_values = args.wc_mask_values
wc_nodata_values = args.wc_nodata_values

input_chm = f"/gpfs/glad1/Theo/Data/Lidar/CHMs_raw/1_Combined_CHMs/{survey}_CHM.tif"
output_folder = f"/gpfs/glad1/Theo/Data/Lidar/CHM_testing/{survey}"
data_folders = ["/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Canopy/Canopy.shp", 
    "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Powerlines", 
    "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Manual_powerlines", 
    "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Water", 
    "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Landsat", 
    "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Slope_errors/Slope_errors.shp",
    "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/WorldCover"]
crs = "EPSG:3857"
pixel_size = 4.77731426716

if slope_mask == True:
    output_tiff = os.path.join(output_folder, f"ndvi{ndvi_threshold}_height{height_threshold}_{survey}_CHM_cleaned.tif")
else:
    output_tiff = f"{os.path.join(output_folder, survey)}_CHM_cleaned.tif"

# Clean the CHM
clean_chm(input_chm, output_tiff, data_folders, crs, pixel_size, buffer_size, save_temp, slope_mask, man_pwl, man_slp, height_threshold, ndvi_threshold, water_mask_values, wc_mask_values, wc_nodata_values)