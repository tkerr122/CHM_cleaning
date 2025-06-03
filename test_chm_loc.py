from All_clean_CHM import *

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
    lat1, lon1, _ = transform.TransformPoint(x_left, y_top)
    lat2, lon2, _ = transform.TransformPoint(x_left, y_bottom)
    lat3, lon3, _ = transform.TransformPoint(x_right, y_bottom)
    lat4, lon4, _ = transform.TransformPoint(x_right, y_top)
    
    print(f"X left, Y top: {lat1, lon1}\nX left, Y bottom: {lat2, lon2}\nX right, Y bottom: {lat3, lon3}\nX right, Y top: {lat4, lon4}")
    
    # Extract tile names
    def get_tile_name(lat, lon):
        lat_dir = "N" if lat > 0 else "S"
        lon_dir = "E" if lon > 0 else "W"
        
        latitude = (int(lat) // 10) * 10 + 10 
        longitude = (int(abs(lon)) // 10) * 10 + 10
        
        lon_str = f"0{longitude}" if longitude < 100 else f"{longitude}"
        return f"{latitude}{lat_dir}_{lon_str}{lon_dir}"
    
    # Get tile names
    tile1 = get_tile_name(lat1, lon1)
    tile2 = get_tile_name(lat2, lon2)
    tile3 = get_tile_name(lat3, lon3)
    tile4 = get_tile_name(lat4, lon4)
    
    tiles = sorted(set([tile1, tile2, tile3, tile4]))
        
    return tiles
    
worldcover, _, _, _, _ = get_raster_info("/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/WorldCover/ESA_WorldCover_10m_2021_v200_N18W156_Map.tif")
get_chm_loc2(worldcover)

chm, _, _, _, _ = get_raster_info("/gpfs/glad1/Theo/Data/Lidar/CHM_testing/WY_GrandTeton/WY_GrandTeton_CHM_cleaned.tif")
get_chm_loc2(chm)

"""
Read in ESA_WorldCover_10m_2021_v200_N18W156_Map.tif...
X left: -156.0
Y bottom: 18.0

Read in WY_GrandTeton_CHM_cleaned.tif...
X left, Y bottom: (43.53377210455333, -110.95530913645823)



/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/WorldCover/ESA_WorldCover_10m_2021_v200_N42W114_Map.tif
/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/WorldCover/ESA_WorldCover_10m_2021_v200_N42W111_Map.tif
/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/WorldCover/ESA_WorldCover_10m_2021_v200_N42W108_Map.tif
/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/WorldCover/ESA_WorldCover_10m_2021_v200_N42W105_Map.tif
/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/WorldCover/ESA_WorldCover_10m_2021_v200_N42W102_Map.tif
/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/WorldCover/ESA_WorldCover_10m_2021_v200_N42W099_Map.tif
/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/WorldCover/ESA_WorldCover_10m_2021_v200_N42W096_Map.tif
/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/WorldCover/ESA_WorldCover_10m_2021_v200_N42W093_Map.tif


"""