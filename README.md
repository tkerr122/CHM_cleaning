Python workflow for removing various errors from Canopy Height models using GDAL, numpy, and geopandas. 

ALL_clean_CHM.py contains all the cleaning functions as well as the capability to loop through an input folder containing raw CHMs and cleaning them.
Clean_CHM.py is a script designed to be run from the command-line, to clean an individual CHM.
Test_NDVI.py doesn't clean a CHM but creates an NDVI raster covering the extent of the input CHM.

The general overview of the cleaning process is:
 - User has a raster CHM that contains some or a combination of 3 issues: powerlines, water, and slope errors (CHM has a height value in areas it shouldn't as a result of those 3 factors).
 - The script requires data folders containing powerline vector files (such as OSM data), water raster files, and NDVI raster files. The user can also digitize their own vector powerlines, as well as vector polygons identifying areas for slope masking.
 - The slope masking process is iterative, meaning the user will specify a height threshold and an NDVI threshold. Pixels with values above the height threshold are considered. Their NDVI value is then calculated and compared with the NDVI threshold value. If the pixel's NDVI value is higher than the threshold, it is retained in the final CHM. If it is below, it is dropped (pixel value set to 0). Any pixels below the height threshold are not considered in slope masking and their value is retained in the final CHM.
 - The user can input the name of the raw CHM to the Clean_CHM.py script along with various flags in order to clean the CHM.
 - These flags include:
      - [-s: survey name] [-bs: powerline buffer size] [-st: save intermediate temp rasters] [-sm: mask out slope errors] [-mp: use a manually collected vector powerline layer] [-ms: use a manually collected vector slope mask layer] [-ht: height threshold for slope masking (values above this number are considered)] [-nt: NDVI threshold for slope masking (value above this are kept)] [-wmv: list of pixel values from the water raster to use when masking water]
