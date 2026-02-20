# Theo Kerr on 3/7/2025
# For use on linux cluster "gdalenv" conda env

# Imports/env settings 
from All_clean_CHM import *
import argparse
gdal.UseExceptions()

"""I have written this script to be a command-line utility for cleaning a CHM of powerlines, water, 
and ifdesired, slope errors, using specific values for height and NDVI thresholds. 
====================================================================================================
-s option: survey name.
-bs option: desired buffer size for powerlines, in meters. Defaults to 50.
-st option: whether or not to save the temp rasters.
-mp option: whether or not to use a manual powerline file for additional powerline masking.
-slp option: threshold for slope masking
-grt option: threshold for greenred masking
-bt option: threshold for building masking

Assumes the following input variables are hardcoded:
 - input_chm
 - data_folders
 - output_folder
 - crs
 - pixel_size
"""

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Script for cleaning single CHM")
    parser.add_argument("-s", "--survey", type=str, help="Survey name", required=True)
    parser.add_argument("-bs", "--buffer-size", type=int, default=50, help="Buffer size")
    parser.add_argument("-st", "--save-temp", action="store_true", help="Save temp dir")
    parser.add_argument("-mp", "--man-pwl", action="store_true", help="Buffer manual powerlines")
    parser.add_argument("-slp", "--slope-threshold", type=int, default=None, help="Cutoff for slope")
    parser.add_argument("-grt", "--greenred-threshold", type=int, default=135, help="Cutoff for greenred")
    parser.add_argument("-bt", "--building-threshold", type=int, default=30, help="Cutoff for building mask")

    # Parse arguments
    args = parser.parse_args()

    # Set up variables
    survey = args.survey
    buffer_size = args.buffer_size
    save_temp = args.save_temp
    man_pwl = args.man_pwl
    slope_threshold = args.slope_threshold
    greenred_threshold = args.greenred_threshold
    building_threshold = args.building_threshold

    input_chm = f"/gpfs/glad1/Theo/Data/Lidar/CHMs_raw/1_Combined_CHMs/{survey}_CHM.tif"
    output_folder = f"/gpfs/glad1/Theo/Data/Lidar/CHM_testing/{survey}"
    data_folders = ["/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Canopy/Canopy.shp", 
        "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Powerlines", 
        "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Manual_powerlines", 
        "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/WorldCover",
        "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Planet_tiles",
        "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Building_mask_2022",
        "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Slope"]
    crs = "EPSG:3857"
    pixel_size = 4.77731426716

    if slope_threshold:
        output_tiff = os.path.join(output_folder, f"{slope_threshold}slp_{survey}_CHM_cleaned.tif")
    else:
        output_tiff = os.path.join(output_folder, f"{survey}_CHM_cleaned.tif")

    # Clean the CHM
    clean_chm(input_chm, output_tiff, data_folders, crs, pixel_size, buffer_size, save_temp, man_pwl, greenred_threshold, building_threshold, slope_threshold)

if __name__ == "__main__":
    main()