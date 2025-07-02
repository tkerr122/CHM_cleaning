from All_clean_CHM import *

# Get loc function will work for planet tiles

input_chm = "/gpfs/glad1/Theo/Data/Lidar/CHMs_raw/1_Combined_CHMs/OH_Statewide_CHM.tif"
chm = gdal.Open(input_chm)
survey = get_chm_survey(input_chm)[0]
state = get_chm_survey(input_chm)[1]
tiles = get_chm_loc(chm)[0]
wc_tiles = get_chm_loc(chm)[1]

