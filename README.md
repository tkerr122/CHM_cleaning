Python workflow for removing various errors from Canopy Height models using GDAL, numpy, and geopandas. 

ALL_clean_CHM.py contains all the cleaning functions as well as the capability to loop through an input folder containing raw CHMs and cleaning them.
Clean_CHM.py is a script designed to be run from the command-line to clean an individual CHM.
Test_NDVI.py doesn't clean a CHM but creates an NDVI raster covering the extent of the input CHM.

The general overview of the cleaning process is:
 - User has a raster CHM that contains some or a combination of 3 issues: powerlines, water, and slope errors (CHM has a height value in areas it shouldn't as a result of those 3 factors).
 - The script requires data folders containing powerline vector files (such as OSM data), water raster files, NDVI raster files, and WorldCover raster files. The user can also digitize their own vector powerlines, as well as vector polygons identifying areas for slope masking.
 - The slope masking process is two-fold and iterative. The first workflow uses the -sm (slope mask) flag, meaning the user will also specify a height threshold and an NDVI threshold. Pixels with values above the height threshold are considered. Their NDVI value is then calculated and compared with the NDVI threshold value. If the pixel's NDVI value is higher than the threshold, it is retained in the final CHM. If it is below, it is dropped (pixel value set to 0). Any pixels below the height threshold are not considered in slope masking and their value is retained in the final CHM. This process covers the entire extent of the CHM.
 - The second workflow uses the -ms (manual slope) flag, meaning the user will have digitized a manual slope file which will be used to identify areas for slope masking. These areas are then compared to the WorldCover image and the pixels in areas identified for masking (e.g. where the land cover is bare ground, snow and ice, etc.) are set to 0 or 255 (nodata) depending on user choice. 
 - A side note for the slope masking workflow: if the user wishes to consider the entire extent of the CHM, they can just digitize the extent into the manual slope mask file.
 - The user can input the name of the raw CHM to the Clean_CHM.py script along with various flags in order to clean the CHM.
 - These flags include:
      [-s: survey name] 
      [-bs: powerline buffer size] 
      [-st: save intermediate temp rasters] 
      [-sm: mask out slope errors] 
      [-mp: use a manually collected vector powerline layer] 
      [-ms: use a manually collected vector slope mask layer] 
      [-ht: height threshold for slope masking (values above this number are considered)] 
      [-wmv: list of pixel values from the water raster to use when masking water]
      [-wcmv option: list of pixel values from worldcover image for masking]

Dataset sources: 
- WorldCover dataset: https://github.com/ESA-WorldCover/esa-worldcover-datasets
- Power line dataset: https://overpass-turbo.eu/

Dataset info:
- Labels for worldcover:
    # 10 Trees covered area
    # 20 Shrub covered area
    # 30 Grassland
    # 40 Cropland
    # 50 Built-up
    # 60 Bare areas
    # 70 Snow and/or ice cover
    # 80 Permament water bodies
    # 90 Herbaceous wetland
    # 95 Mangroves
    # 100 Lichens and mosses