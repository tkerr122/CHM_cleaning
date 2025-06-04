# Theo Kerr
# For use on linux cluster "gdalenv" conda env

# Imports/env settings 
import numpy as np
import geopandas as gpd
import os, shutil, math
from osgeo import gdal, osr, ogr
from tqdm import tqdm
gdal.UseExceptions()

# Define custom errors
class InvalidSurvey(Exception):
    pass


def get_chm_survey(chm_path):
    """Parses given CHM path for survey and state.

    Args:
        chm_path (str): path to CHM, assumed to be in format "path/to/chm/[state_abbr]_[survey]_CHM.tif".

    Returns:
        tuple: tuple containing survey name, and state.
    """
    chm_raw = os.path.splitext(os.path.basename(chm_path))[0]
    survey_raw = chm_raw.rsplit("_CHM")[0]
    state = survey_raw.rsplit("_")[0]
    
    return survey_raw, state 

def get_chm_loc(chm):
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
        
    tiles = get_tile_name(lat_min, lat_max, lon_min, lon_max)
    tiles = sorted(set(tiles))
        
    return tiles

def get_worldcover_loc(chm):
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
                tile_names.append(tile_name)
        
        return tile_names
        
    tiles = get_tile_name(lat_min, lat_max, lon_min, lon_max)
    tiles = sorted(set(tiles))
        
    return tiles

def get_raster_info(raster_path):
    """Opens a raster at the given path.

    Args:
        raster_path (str): path to a raster dataset.

    Returns:
        tuple: returns a GDAL raster, number of columns, number of rows, the geotransform, and the projection.
    """
    ds = gdal.Open(raster_path)
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    transform = ds.GetGeoTransform()
    projection = ds.GetSpatialRef()
    
    print(f"Read in {os.path.basename(raster_path)}...")
    
    return ds, xsize, ysize, transform, projection

def extract_polygon(input_shp, survey, output_folder):
    """Reads in a shapefile and extracts the polygons with attribute "survey" == given survey value, creating a new output GeoJSON file.

    Args:
        input_shp (str): path to shapefile.
        survey (str): string for survey name (i.e. "AZ_BlackRock").
        output_folder (str): path to output folder for the new GeoJSON file.

    Raises:
        InvalidSurvey: custom error to make sure that the canopy shp indeed has polygons for the indicated survey, or that the survey name doesn't have a typo.

    Returns:
        str: path to the new canopy GeoJSON.
    """
    # Load in canopy shapefile and mask to survey
    shp = gpd.read_file(input_shp)
    polygon = shp[shp["Survey"] == survey]
    polygon_basename = os.path.splitext(os.path.basename(input_shp))[0]
    path = os.path.join(output_folder, f"{polygon_basename}_{survey}.geojson")
    if polygon.empty:
        print(f"\nThe survey name \"{survey}\" is incorrect or doesn't exist\n")
        raise InvalidSurvey(survey)
    
    # Write canopy geojson to file, if it hasn't already been done
    if os.path.isfile(path) == False:
        polygon.to_file(path, driver="GeoJSON")
        print(f"Created {os.path.basename(path)}...")
    else:
        print(f"\"{os.path.basename(path)}\" exists, saving path...")
    
    return path

def crop_raster(raster_path, output_folder, crs, pixel_size, cutline):
    """Uses the GDAL Warp function to reproject the given raster to given crs and pixel size, and crop to the given cutline. 

    Args:
        raster_path (str): path to a raster dataset.
        output_folder (str): folder for the output dataset.
        crs (str): string for a crs, in the format "EPSG:3857" for example.
        pixel_size (float): desired pixel size, in destination crs units.
        cutline (str): path to a GeoJSON cutline file.

    Returns:
        str: path to output warped raster.
    """
    # Set warp options
    if type(raster_path) == list and len(raster_path) != 1:
        raster_basename = os.path.splitext(os.path.basename(raster_path[0]))[0]
        dst_ds = f"{os.path.join(output_folder, raster_basename)}_cropped_merged.tif"
    elif type(raster_path) == list and len(raster_path) == 1:
        raster_basename = os.path.splitext(os.path.basename(raster_path[0]))[0]
        dst_ds = f"{os.path.join(output_folder, raster_basename)}_cropped.tif"
    else: 
        raster_basename = os.path.splitext(os.path.basename(raster_path))[0]
        dst_ds = f"{os.path.join(output_folder, raster_basename)}_cropped.tif"
    
    # Crop the raster, if it hasn't already been done
    if os.path.isfile(dst_ds) == False:
        print(f"Cropping {os.path.basename(dst_ds)}:")
        gdal.Warp(dst_ds, raster_path, format="GTiff", dstSRS=crs, xRes=pixel_size, yRes=pixel_size, cutlineDSName=cutline, cropToCutline=True, warpOptions=["COMPRESS=LZW", "BIGTIFF=YES"], callback=gdal.TermProgress_nocb)
    else:
        print(f"\"{os.path.basename(dst_ds)}\" exists, saving path...")
        
    return dst_ds

def rasterize(input_file, output_tiff, pixel_size, burn_value=None):
    """Uses the GDAL Rasterize Layer function to rasterize the given vector layer to the output tiff path, with the given pixel size. If a burn value is not specified, 
	the polygon value attribute is used.

    Args:
        input_file (str): path to vector dataset.
        output_tiff (str): path to output raster dataset.
        pixel_size (float): desired pixel size.
        burn_value (int, optional): desired output pixel value. Defaults to 1.
    """
    # Check if file exists
    if os.path.isfile(output_tiff) == False:
        # Open dataset
        dataset = ogr.Open(input_file)
        layer = dataset.GetLayer() 
        
        # Define raster properties
        x_min, x_max, y_min, y_max = layer.GetExtent()
        x_res = int((x_max - x_min) / pixel_size)
        y_res = int((y_max - y_min) / pixel_size)
        target_ds = gdal.GetDriverByName("GTiff").Create(output_tiff, x_res, y_res, 1, gdal.GDT_Byte, options=["COMPRESS=LZW", "BIGTIFF=YES"])
        target_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
        
        # Set projection
        srs = layer.GetSpatialRef()
        target_ds.SetProjection(srs.ExportToWkt())
        
        # Set band nodata value
        band = target_ds.GetRasterBand(1)
        band.SetNoDataValue(255)

        # Rasterize dataset
        print(f"Rasterizing {os.path.basename(input_file)}:")
        if burn_value is not None:
            gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[burn_value], callback=gdal.TermProgress_nocb)
        else:
            gdal.RasterizeLayer(target_ds, [1], layer, options=["ATTRIBUTE=value"], callback=gdal.TermProgress_nocb)
      
        band = None
        target_ds = None
        dataset = None
        
    else:
        print(f"\"{os.path.basename(output_tiff)}\" exists, not rasterizing")
    
    
def buffer_powerlines(input_file, output_file, crs, pixel_size, buffer_size, cutline, burn_value=1):
    """Reads in given vector dataset and buffers it to given specifications.

    Args:
        input_file (str): path to input vector file.
        output_file (str): path to desired output GeoJSON.
        crs (str): string for the desired CRS, in the format "EPSG:3857".
        pixel_size (float): desired pixel size.
        buffer_size (int): desired size for the buffer.
        cutline (str): path to cutline GeoJSON for cropping.
        burn_value (int, optional): desired output pixel value. Defaults to 1.
    """
    # Check if file exists
    if os.path.exists(output_file) == False:
        # Read in powerline
        powerline = gpd.read_file(input_file)
        powerline = powerline.to_crs(crs)
        
        # Crop to cutline
        cutline = gpd.read_file(cutline)
        cutline = cutline.to_crs(crs)
        powerline_cropped = gpd.clip(powerline, cutline)
        
        # Buffer to specified radius
        powerline_buffer = powerline_cropped.buffer(buffer_size, cap_style="square")
        output_geojson = f"{os.path.splitext(output_file)[0]}.geojson"
        powerline_buffer.to_file(output_geojson, driver="GeoJSON")
        
        # Rasterize the buffer
        rasterize(output_geojson, output_file, pixel_size, burn_value)
    
        # Remove temporary geojson buffer
        os.remove(output_geojson)
        
    else:
        print(f"\"{os.path.basename(output_file)}\" exists, not buffering")

def mask_powerlines(chm_array, powerlines_array):
    """Takes in an array for the CHM and the powerlines (must be same dimensions) and sets CHM values to 0 where there are powerlines.

    Args:
        chm_array (np.array): array for the CHM.
        powerlines_array (np.array): array for the powerlines (1 for presence).

    Returns:
        np.array: cleaned CHM array.
    """
    condition_mask = (powerlines_array == 1) & (powerlines_array != 255)
    chm_cleaned = np.where(condition_mask, 0, chm_array)
    chm_cleaned[chm_array == 255] = 255
    print("Cleaned CHM Powerlines...")
    
    return chm_cleaned

def mask_water(chm_array, water_array, water_mask_values):
    """Takes in an array for the CHM and the water mask (must be same dimensions) and sets CHM values to 0 where there is water.

    Args:
        chm_array (np.array): array for the CHM.
        water_array (np.array): array for the water mask.
        water_mask_values (list): list of pixel values to retain as "True" for the water mask.

    Returns:
        np.array: cleaned CHM array.
    """
    water_mask = np.isin(water_array, water_mask_values).astype(int)
    water_mask[water_mask == 0] = 255
    water_mask[water_mask == 1] = 1
    condition_mask = (water_mask == 1) & (water_mask != 255)
    chm_cleaned = np.where(condition_mask, 0, chm_array)
    print("Cleaned CHM Water...")
    
    return chm_cleaned

def calc_ndvi(nir_band, red_band, output_band, x, y, cols, rows, threshold_value, mask): 
    """Takes in nir and red bands from an image raster, a desired NDVI threshold value (on 8-bit scale), and the position/size of the image raster.
    Calculates NDVI for that position in the image, and if mask == True, thresholds it to the specified value and writes the mask array to the specified band.
    If mask == False, writes the NDVI array to the output band.  

    Args:
        nir_band (gdal.RasterBand): NIR band from image raster.
        red_band (gdal.RasterBand): Red band from image raster.
        output_band (gdal.RasterBand): desired band of output raster to write the NDVI mask to threshold_value (int): 8-bit scaled NDVI threshold value.
        x (int): x position in the image raster.
        y (int): y position in the image raster.
        cols (int): x size of the image raster.
        rows (int): y size of the image raster.
        threshold_value (int): 8-bit scaled NDVI threshold value.
        mask (bool): whether to create an NDVI mask based on given threshold value.
    """
    # Read in bands as numpy array
    nir_32 = nir_band.ReadAsArray(x, y, cols, rows).astype(np.float32)
    red_32 = red_band.ReadAsArray(x, y, cols, rows).astype(np.float32)
        
    # Calculate NDVI
    numerator = np.subtract(nir_32, red_32)
    denominator = np.add(nir_32, red_32)
    epsilon = 1e-6
    denominator[denominator == 0] = epsilon 
    result = np.divide(numerator, denominator)
    
    # Remove out of bounds areas
    result[result == -0] = 0
    
    # Scale to 8-bit and mask to threshold, if specified
    ndvi = np.multiply((result + 1), (2**7 - 1))
    if mask == True: 
        ndvi = np.where(ndvi > threshold_value, 1, 255)
    
    # Write to raster
    output_band.WriteArray(ndvi, x, y)
    del ndvi
        
def calc_ndvi_by_block(input_image, output_folder, threshold_value=None, mask=True):
    """Uses array indexing to compute NDVI by block for a given landsat image using the calc_ndvi function.
    
    Args:
        input_image (str): path to landsat image raster.
        output_folder (str): path to desired output folder.
        threshold_value (int, optional): 8-bit scaled NDVI threshold value. Defaults to None.
        mask (bool, optional): whether to create an NDVI mask based on given threshold value. Defaults to True.

    Returns:
        str: path to output NDVI raster.
    """
    # Read in landsat image
    landsat_image, xsize, ysize, geotransform, srs = get_raster_info(input_image)
    nir = landsat_image.GetRasterBand(2)
    red = landsat_image.GetRasterBand(1)
    
    # Set block size
    x_block_size = 256
    y_block_size = 160
    
    # Create new raster
    if threshold_value is not None:
        output_path = os.path.join(output_folder, f"ndvi_{threshold_value}.tif")
    else:
        output_path = os.path.join(os.path.dirname(output_folder), f"ndvi.tif")
    output = gdal.GetDriverByName("GTiff").Create(output_path, xsize, ysize, 1, gdal.GDT_Byte, options=["COMPRESS=LZW", "BIGTIFF=YES"])
    output_band = output.GetRasterBand(1)
    output_band.SetNoDataValue(255)
    output.SetGeoTransform(geotransform)
    output.SetProjection(srs.ExportToWkt())
    
    # Mask NDVI
    total_blocks = (xsize // x_block_size + 1) * (ysize // y_block_size + 1)
    progress_bar = tqdm(total=total_blocks, desc="Progress", unit="block")
    
    for y in range(0, ysize, y_block_size):
        rows = min(y_block_size, ysize - y)  # Handles edge case for remaining rows
        for x in range(0, xsize, x_block_size):
            cols = min(x_block_size, xsize - x)  # Handles edge case for remaining cols
            calc_ndvi(nir, red, output_band, x, y, cols, rows, threshold_value, mask)
            progress_bar.update(1)
     
    progress_bar.close()
    output_band = None
    output = None
    landsat_image = None
    
    return output_path

def mask_slope(chm_array, ndvi_array, height_threshold, slope_8bit=None):
    """Takes in an array for the CHM and the NDVI mask (must be same dimensions) and sets CHM values to 0 where the CHM height is above the given threshold, masking with NDVI mask.
    If slope_mask array is provided, will only mask the portions of the CHM that overlap the slope_mask array, using those values as the height threshold.

    Args:
        chm_array (np.array): array for the CHM.
        ndvi_array (np.array): array for the NDVI.
        height_threshold (int): 8-bit scaled height threshold value.
        slope_8bit (np.array, optional): array of manual slope errors. Defaults to None.

    Returns:
        np.array: cleaned CHM array.
    """
    ndvi_mask = (ndvi_array == 255)
    if slope_8bit is not None:
        ground_mask = (slope_8bit == 0)
        
        # Set areas designated as ground to 0
        chm_array = np.where(ground_mask, 0, chm_array)
        threshold_mask = (chm_array > slope_8bit) & ndvi_mask
       
    else:
        threshold_mask = (chm_array > height_threshold) & ndvi_mask
    chm_cleaned = np.where(threshold_mask, 0, chm_array)
    chm_cleaned[chm_array == 255] = 255
    print("Cleaned CHM slope errors...")
    
    return chm_cleaned

def preprocess_data_layers(input_chm, temp, data_folders, crs, pixel_size, buffer_size=50, sm=False, man_pwl=False, man_slp=False):
    """Creates a temporary folder for all intermediate data layers and performs preprocessing on them such as buffering, cropping, and generating file paths as specified.
    If the temp folder already exists, will just return the filenames of the previously created layers. 
    IMPORTANT: if performing multiple iterations of CHM cleaning, keep in mind that persisting the temp folder also persists the layers, so for example the powerline buffer raster will
    be read in as is, it will not be re-buffered. 

    Args:
        input_chm (str): path to input chm.
        temp (str): path to temp directory.
        data_folders (list): list of paths to the relevant data folders for the canopy shapefile, powerline masks, manual powerline masks, water images, landsat images, and slope errors shapefile.
        crs (str): string for the desired CRS, in the format "EPSG:3857" for example.
        pixel_size (float): desired pixel size for reprojection, in destination crs units.
        buffer_size (int, optional): desired buffer size for powerlines, in meters. Defaults to 50.
        sm (bool, optional): whether or not to mask out the slope errors. Defaults to False.
        man_pwl (bool, optional): whether or not to use a manual powerline file for additional powerline masking. Defaults to False.
        man_slp (bool, optional): whether or not to use a manual slope errors shapefile for slope masking. Defaults to False. 

    Raises:
        InvalidSurvey: depending on the condition, prints message stating how the input survey is invalid.

    Returns:
        tuple: tuple containing chm_cropped_path, powerlines_cropped_path, water_cropped_path, landsat_cropped_path, man_pwl_cropped_path, and man_slp_cropped_path.
    """
    # Check if temp folder is already populated
    if os.path.isdir(temp) == False:
        os.makedirs(temp, exist_ok=True)
    
    # Read in CHM and get info
    chm = gdal.Open(input_chm)
    survey = get_chm_survey(input_chm)[0]
    state = get_chm_survey(input_chm)[1]
    tiles = get_chm_loc(chm)

    print(f"Got CHM info: \tsurvey: {survey}\t state: {state}\t tile: {tiles}")
    
    chm = None
    
    # Generate paths to corresponding mask layers and preprocess
    try:
        # Create canopy mask
        cutline = extract_polygon(data_folders[0], survey, temp)
        
        # Powerlines
        powerlines_path = os.path.join(data_folders[1], f"{state}_powerlines.geojson")
        if os.path.isfile(powerlines_path) == False:
            print(f"\nError: \"{state}_powerlines.geojson\" doesn't exist.\n")
            raise InvalidSurvey
        
        output_powerlines = os.path.join(temp, f"{state}_powerlines_buffer.tif")
        buffer_powerlines(powerlines_path, output_powerlines, crs, pixel_size, buffer_size, cutline)
        
        # Manual powerlines
        if man_pwl == True:
            man_pwl_path = os.path.join(data_folders[2], f"{state}_man_pwl.geojson")
            if os.path.isfile(man_pwl_path) == False:
                print(f"\nError: Manual powerline file \"{state}_man_pwl.geojson\" doesn't exist.\n")
                raise InvalidSurvey
            
            output_man_pwl = os.path.join(temp, f"{state}_man_pwl_buffer.tif")
            buffer_powerlines(man_pwl_path, output_man_pwl, crs, pixel_size, buffer_size, cutline)
        
        # Water and Landsat images
        water_path = []
        landsat_path = []
        for tile in tiles: 
            w_path = os.path.join(data_folders[3], f"{tile}_water.tif")
            n_path = os.path.join(data_folders[4], f"Hansen_GFC-2023-v1.11_last_{tile}.tif")
            
            if os.path.isfile(w_path) == False:
                print(f"\nError: \"{tile}_water.tif\" doesn't exist.\n")
                raise InvalidSurvey
            
            if os.path.isfile(n_path) == False:
                print(f"\nError: \"Hansen_GFC-2023-v1.11_last_{tile}.tif\" doesn't exist.\n")
                raise InvalidSurvey
            
            water_path.append(w_path)
            landsat_path.append(n_path)
        
    except InvalidSurvey:
        shutil.rmtree(temp)
        exit()

    # If manual slope mask is specified, rasterize this as another mask layer
    if man_slp == True:
        # Extract slope errors for current survey
        slope_errors = extract_polygon(data_folders[5], survey, temp)
        
        # Rasterize the slope errors
        slope_mask_path = os.path.join(temp, f"{survey}_slope_mask.tif")
        rasterize(slope_errors, slope_mask_path, pixel_size)
        
    # Crop the rasters to the extent of the canopy mask layer
    chm_cropped_path = crop_raster(input_chm, temp, crs, pixel_size, cutline)
    powerlines_cropped_path = crop_raster(output_powerlines, temp, crs, pixel_size, cutline)
    water_cropped_path = crop_raster(water_path, temp, crs, pixel_size, cutline)
    if sm == True:
        landsat_cropped_path = crop_raster(landsat_path, temp, crs, pixel_size, cutline)
    if man_pwl == True:
        man_pwl_cropped_path = crop_raster(output_man_pwl, temp, crs, pixel_size, cutline)
    if man_slp == True: 
        man_slp_cropped_path = crop_raster(slope_mask_path, temp, crs, pixel_size, cutline)
                
    # Return the filepaths
    if "landsat_cropped_path" not in locals():
        landsat_cropped_path = None
    if "man_pwl_cropped_path" not in locals():
        man_pwl_cropped_path = None
    if "man_slp_cropped_path" not in locals():
        man_slp_cropped_path = None
        
    return chm_cropped_path, powerlines_cropped_path, water_cropped_path, landsat_cropped_path, man_pwl_cropped_path, man_slp_cropped_path

def clean_chm(input_chm, output_tiff, data_folders, crs, pixel_size, buffer_size=50, save_temp=False, sm=False, man_pwl=False, man_slp=False, height_threshold=None, ndvi_threshold=None, water_mask_values=[2, 3, 4, 5, 6, 7, 8, 11]):
    """CHM cleaning workflow using all the previously defined functions. Users can specify the desired powerline buffer, whether to save the temporary output rasters, mask the slope errors, use manual powerline and slope layers for maksing, desired thresholds if they do mask slope, and a list of pixel values to retain for water masking.
    Steps:
    1. Gets raster information for the CHM.
    2. Creates a canopy cutline according to the survey.
    3. Crops the rasters.
    4. Sets up blank output raster.
    5. Masks powerlines, water, and slope (if speficied).
        - If man_pwl == True, will also buffer and mask the CHM to the manual powerlines file.
        - If sm == True, will mask slope across entire extent of CHM.
        - If man_slp == True, will mask slope across extent of manual slope errors shapefile.
    6. Writes the new raster (and deletes the temp files, if specified).

    Args:
        input_chm (str): path to input CHM.
        output_tiff (str): path to output raster.
        data_folders (list): list of paths to the relevant data folders for the canopy shapefile, powerline masks, water masks, and landsat images, respectively
        crs (str): string for the desired CRS, in the format "EPSG:3857" for example.
        pixel_size (float): desired pixel size for reprojection, in destination crs units.
        buffer_size (int, optional): desired buffer size for powerlines, in meters. Defaults to 50.
        save_temp (bool, optional): whether or not to save the temp rasters. Defaults to False.
        sm (bool, optional): whether or not to mask out the slope errors. Defaults to False.
        man_pwl (bool, optional): whether or not to use a manual powerline file for additional powerline masking. Defaults to False.
        man_slp (bool, optional): whether or not to use a manual slope errors shapefile for slope masking. Defaults to False. 
        height_threshold (int, optional): height threshold (values above this number are considered) used for slope masking. Defaults to None.
        ndvi_threshold (int, optional): NDVI threshold (values above this are kept). Defaults to None.
        water_mask_values (list, optional): list of pixel values from water image to use when masking water (i.e. if value in list, set CHM value to 0). Defaults to [2, 3, 4, 5, 6, 7, 8, 11]
    """
    # Start message
    print(f" PROCESSING CHM {os.path.basename(input_chm)} ".center(100, "*"))
    
    # Set up temp directory for intermediate files
    temp = os.path.join(os.path.dirname(output_tiff), "temp")
    
    # Preprocess the data layers
    chm_cropped_path, powerlines_cropped_path, water_cropped_path, landsat_cropped_path, man_pwl_cropped_path, man_slp_cropped_path = preprocess_data_layers(input_chm, temp, data_folders, crs, pixel_size, buffer_size, sm, man_pwl, man_slp)
                
    # Create output blank raster
    chm_cropped, c_xsize, c_ysize, c_geotransform, c_srs = get_raster_info(chm_cropped_path)
    output = gdal.GetDriverByName("GTiff").Create(output_tiff, c_xsize, c_ysize, 1, gdal.GDT_Byte, options=["COMPRESS=LZW", "BIGTIFF=YES"])
    output_band = output.GetRasterBand(1)
    output_band.SetNoDataValue(255)
    output.SetGeoTransform(c_geotransform)
    output.SetProjection(c_srs.ExportToWkt())
    print("Created blank raster...")
    
    # Mask CHM by powerlines
    chm_8bit = chm_cropped.GetRasterBand(1).ReadAsArray(0, 0, c_xsize, c_ysize).astype(np.uint8)
    powerlines_cropped, p_xsize, p_ysize, _, _ = get_raster_info(powerlines_cropped_path)
    powerlines_8bit = powerlines_cropped.GetRasterBand(1).ReadAsArray(0, 0, p_xsize, p_ysize).astype(np.uint8)
    chm_cleaned = mask_powerlines(chm_8bit, powerlines_8bit)
    
    chm_cropped = None
    chm_8bit = None
    powerlines_cropped = None
    powerlines_8bit = None
    
    # Mask CHM by manual powerlines (if specified)
    if man_pwl == True:
        man_pwl_cropped, mp_xsize, mp_ysize, _, _ = get_raster_info(man_pwl_cropped_path)
        man_pwl_8bit = man_pwl_cropped.GetRasterBand(1).ReadAsArray(0, 0, mp_xsize, mp_ysize).astype(np.uint8)
        chm_cleaned = mask_powerlines(chm_cleaned, man_pwl_8bit)
        
        man_pwl_cropped = None
        man_pwl_8bit = None
        
    # Mask CHM by water
    water_cropped, w_xsize, w_ysize, _, _ = get_raster_info(water_cropped_path)
    water_8bit = water_cropped.GetRasterBand(1).ReadAsArray(0, 0, w_xsize, w_ysize).astype(np.uint8)
    chm_cleaned = mask_water(chm_cleaned, water_8bit, water_mask_values)
    
    water_cropped = None
    water_8bit = None
    
    # Mask CHM by slope (if specified)
    if sm == True:
        # Create NDVI mask
        ndvi_path = calc_ndvi_by_block(landsat_cropped_path, temp, ndvi_threshold)
        ndvi_cropped, n_xsize, n_ysize, _, _ = get_raster_info(ndvi_path)
        ndvi_8bit = ndvi_cropped.GetRasterBand(1).ReadAsArray(0, 0, n_xsize, n_ysize).astype(np.uint8)
        
        # Read in manual slope mask raster (if specified), use to clean slope
        if man_slp == True:
            slope_cropped, s_xsize, s_ysize, _, _ = get_raster_info(man_slp_cropped_path)
            slope_8bit = slope_cropped.GetRasterBand(1).ReadAsArray(0, 0, s_xsize, s_ysize).astype(np.uint8)
            
            chm_cleaned = mask_slope(chm_cleaned, ndvi_8bit, height_threshold, slope_8bit)
        
        else:
            # Clean slope errors (without using manual slope mask raster)
            chm_cleaned = mask_slope(chm_cleaned, ndvi_8bit, height_threshold)
    
        ndvi_cropped = None
        ndvi_8bit = None
        slope_cropped = None
        slope_8bit = None
        
    # Write cleaned CHM to new raster
    output_band.WriteArray(chm_cleaned)
    print(f"Cleaned CHM written to {output_tiff}")
    print(f" FINISHED PROCESSING CHM {os.path.basename(input_chm)}".center(100, "*"))
    print("\n")
    
    chm_cleaned = None
    output_band = None
    output = None
    
    if save_temp == False:
        shutil.rmtree(temp)
    
def main(input_folder, output_folder, data_folders, crs, pixel_size):
    # Create the output folder if it doesn't already exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Loop through input folder
    for file in os.listdir(input_folder):
        # Generate i/o filenames
        input_chm = os.path.join(input_folder, file)
        output_chm = os.path.splitext(file)[0]
        output_tiff = f"{os.path.join(output_folder, output_chm)}_cleaned.tif"

        # Clean the CHM
        if os.path.exists(output_tiff):
            print(f"CHM {file} already exists, skipping...")
            continue
        else:
            clean_chm(input_chm, data_folders, output_tiff, crs, pixel_size)

if __name__ == "__main__":
    input_folder = "/gpfs/glad1/Theo/Data/Lidar/CHMs_raw/1_Combined_CHMs/For_cleaning"
    output_folder = "/gpfs/glad1/Theo/Data/Lidar/Cleaned_CHMs"
    data_folders = ["/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Canopy/Canopy.shp", 
                    "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Powerlines", 
                    "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Manual_powerlines", 
                    "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Water", 
                    "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Landsat", 
                    "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/Slope_errors/Slope_errors.shp", 
                    "/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/WorldCover"]
    crs = "EPSG:3857"
    pixel_size = 4.77731426716
    main(input_folder, output_folder, data_folders, crs, pixel_size)