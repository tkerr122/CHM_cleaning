from All_clean_CHM import *
"""
#TODO: 
//1. Add a step in preprocess_data_layers function for worldcover
//2. Write a mask_worldcover function
//    - Should take input chm array, input worldcover array, mask values to set as 255, and mask values to set as 0
//    - Follow structure of mask_slope using manual slope file
- Modify the input list to be words instead of pixel values
//- Modify so that slope values of 0 are immediately set to 0
//- Add mask_worldcover to clean_chm function
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

output_tiff = f"{os.path.join(output_folder, 'WY_GrandTeton_CHM')}_test_cleaned.tif"

# Read in all layers needed
chm_cropped_path, powerlines_cropped_path, water_cropped_path, landsat_cropped_path, man_pwl_cropped_path, man_slp_cropped_path, worldcover_cropped_path = preprocess_data_layers(input_chm, temp, data_folders, crs, pixel_size, man_slp=True)

# Create output blank raster
chm_cropped, c_xsize, c_ysize, c_geotransform, c_srs = get_raster_info(chm_cropped_path)
chm_8bit = chm_cropped.GetRasterBand(1).ReadAsArray(0, 0, c_xsize, c_ysize).astype(np.uint8)
output = gdal.GetDriverByName("GTiff").Create(output_tiff, c_xsize, c_ysize, 1, gdal.GDT_Byte, options=["COMPRESS=LZW", "BIGTIFF=YES"])
output_band = output.GetRasterBand(1)
output_band.SetNoDataValue(255)
output.SetGeoTransform(c_geotransform)
output.SetProjection(c_srs.ExportToWkt())
print("Created blank raster...")

# Mask slope using worldcover
slope_cropped, s_xsize, s_ysize, _, _ = get_raster_info(man_slp_cropped_path)
wc_cropped, wc_xsize, wc_ysize, _, _ = get_raster_info(worldcover_cropped_path)
slope_8bit = slope_cropped.GetRasterBand(1).ReadAsArray(0, 0, s_xsize, s_ysize).astype(np.uint8)
wc_8bit = wc_cropped.GetRasterBand(1).ReadAsArray(0, 0, wc_xsize, wc_ysize).astype(np.uint8)

chm_cleaned = mask_worldcover(chm_8bit, slope_8bit, wc_8bit, [30, 60, 70, 80], [50])

# Write cleaned CHM to new raster
output_band.WriteArray(chm_cleaned)
print(f"Cleaned CHM written to {output_tiff}")

chm_cleaned = None
output_band = None
output = None

'''
Worldcover legend:

wc_mask_values=[60, 70, 80], wc_nodata_values=[50]

    NO_DATA = (0, 'nodata', 'Not sure', 'No Data', np.array([0, 0, 0]))
    TREE = (10, 'tree', 'tree', 'Trees covered area',
            np.array([0, 100, 0]) / 255)
    SHRUB = (20, 'shrub', 'shrub', 'Shrub cover area',
             np.array([255, 187, 34]) / 255)
    GRASS = (30, 'grass', 'grassland', 'Grassland',
             np.array([255, 255, 76]) / 255)
    CROP = (40, 'crop', 'crops', 'Cropland', np.array([240, 150, 255]) / 255)
    BUILT = (50, 'built', 'urban/built-up',
             'Built-up', np.array([250, 0, 0]) / 255)
    BARE = (60, 'bare', 'bare', 'Bare areas', np.array([180, 180, 180]) / 255)
    SNOW_AND_ICE = (70, 'snow', 'snow and ice', 'Snow and/or ice cover',
                    np.array([240, 240, 240]) / 255)
    WATER = (80, 'water', 'water', 'Permanent water',
             np.array([0, 100, 200]) / 255)
    WETLAND = (90, 'wetland', 'wetland (herbaceous)', 'Herbaceous wetland',
               np.array([0, 150, 160]) / 255)
    MANGROVES = (95, 'mangroves', None, 'Mangroves',
                 np.array([0, 207, 117]) / 255)
    LICHENS = (100, 'lichens_mosses', 'Lichen and moss', 'Lichen and moss',
               np.array([250, 230, 160]) / 255)
'''