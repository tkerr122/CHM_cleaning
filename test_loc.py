from All_clean_CHM import *

chm, _, _, _, _ = get_raster_info("/gpfs/glad1/Theo/Data/Lidar/Cleaned_CHMs/CA_SierraNevada_CHM_cleaned.tif")

tiles = get_chm_loc(chm)
print(tiles)