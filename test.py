from All_clean_CHM import *

# Define new red-green calculation functions
# Get loc function will work for planet tiles


def mask_city(chm_array, city_array, redgreen_array):
    """Takes in an array for the CHM and the NDVI mask (must be same dimensions) and sets CHM values to 0 where the CHM height is above the given threshold, masking with NDVI mask.

    Args:
        chm_array (np.array): array for the CHM.
        ndvi_array (np.array): array for the NDVI.
        height_threshold (int): 8-bit scaled height threshold value.

    Returns:
        np.array: cleaned CHM array.
    """
    redgreen_mask = (redgreen_array == 255)
    city_mask = (city_array == 1)
    threshold_mask = redgreen_mask & city_mask
    chm_cleaned = np.where(threshold_mask, 0, chm_array)
    chm_cleaned[chm_array == 255] = 255
    print("Cleaned CHM city errors...")
    
    return chm_cleaned