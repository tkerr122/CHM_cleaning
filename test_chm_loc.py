from All_clean_CHM import *

#TODO: figure out how to fill gaps between edges of bounding box (rasters in between both extremes)


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

chm, _, _, _, _ = get_raster_info("/gpfs/glad1/Theo/Data/Lidar/CHM_testing/WY_GrandTeton/WY_GrandTeton_CHM_cleaned.tif")

worldcover = get_chm_loc2(chm)
print(f"\n{worldcover}")

'''
Read in ESA_WorldCover_10m_2021_v200_N48W111_Map.tif...
X left: -111.0
X right: -108.0
Y top: 51.0
Y bottom: 48.0

Read in MT_Highline_CHM_cleaned.tif...
X left: -112.18395632894597
X right: -104.0305559089358
Y top: 49.0163131152675
Y bottom: 46.63570638327659

'''