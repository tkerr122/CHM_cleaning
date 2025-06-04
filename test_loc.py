from All_clean_CHM import *
"""
#TODO: 
1. Add a step in preprocess_data_layers function for worldcover
2. Write a mask_worldcover function
    - Should take input chm array, input worldcover array, mask values to set as 255, and mask values to set as 0
3. Add mask_worldcover to clean_chm function
"""
input_chm = f"/gpfs/glad1/Theo/Data/Lidar/CHM_testing/WY_GrandTeton/WY_GrandTeton_CHM_cleaned.tif"
output_folder = f"/gpfs/glad1/Theo/Data/Lidar/CHM_testing/WY_GrandTeton"
data_folders = ["/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Canopy/Canopy.shp", 
    "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Powerlines", 
    "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Manual_powerlines", 
    "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Water", 
    "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Landsat", 
    "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Slope_errors/Slope_errors.shp",
    "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/WorldCover"]

crs = "EPSG:3857"
pixel_size = 4.77731426716
temp = os.path.join(output_folder, "temp")

chm_cropped_path, powerlines_cropped_path, water_cropped_path, landsat_cropped_path, man_pwl_cropped_path, man_slp_cropped_path, worldcover_cropped_path = preprocess_data_layers(input_chm, temp, data_folders, crs, pixel_size)
