from All_clean_CHM import *

chm, _, _, _, _ = get_raster_info("/gpfs/glad1/Theo/Data/Lidar/CHMs_raw/1_Combined_CHMs/CA_NorthCoast_CHM.tif")

tiles = get_chm_loc(chm)
print(tiles)