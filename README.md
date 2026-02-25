# CHM Cleaning Pipeline

A Python workflow for removing errors from Canopy Height Models (CHMs) using GDAL, NumPy, and GeoPandas.

---

## Scripts

| Script | Purpose |
|---|---|
| `All_clean_CHM.py` | Core cleaning functions |
| `Clean_CHM.py` | Command-line tool for cleaning a single CHM |
| `Test_greenred.py` | Generates a greenred ratio raster cropped to a CHM's extent |

---

## Recommended Workflow

> **Start with `Test_greenred.py` before cleaning a new survey.** This lets you inspect the greenred raster and determine an appropriate `-grt` threshold for filtering building errors before committing to a full clean.

1. Run `Test_greenred.py -s <survey_name>` to generate the greenred raster
2. Inspect the output to choose a suitable `-grt` threshold
3. Run `Clean_CHM.py` with your chosen flags

---

## What the Pipeline Cleans

The CHM cleaner targets four types of errors — pixels with height values in areas they shouldn't have:

- **Powerlines** — detected using OSM vector data, with an optional manual powerline layer
- **Water** — identified using WorldCover raster data
- **Buildings** — filtered using a greenred ratio derived from Planet imagery, and GLAD built-up area probability layer
- **Slope errors** — optionally masked using a numeric slope threshold

---

## Cleaning Process Overview

### Powerline Masking
Powerline vector files are buffered by a user-specified distance (default: 50 m) and used to null out CHM pixels within that buffer. An optional manually digitized powerline layer can be added for additional coverage.

### Water Masking
WorldCover raster data is used to identify permanent water bodies (class `80`) and null out corresponding CHM pixels.

### Building Masking
Planet imagery is used to calculate a greenred ratio. Pixels with a greenred value below the threshold (`-grt`) are treated as potential building errors and removed.

### Slope Masking
A numeric slope threshold (`-slp`) can be optionally applied. Pixels above the specified slope value are candidates for masking.

---

## `Clean_CHM.py` — CLI Flags

```
python Clean_CHM.py -s <survey_name> [options]
```

| Flag | Long form | Default | Description |
|---|---|---|---|
| `-s` | `--survey` | *(required)* | Survey name |
| `-bs` | `--buffer-size` | `50` | Powerline buffer size in meters |
| `-st` | `--save-temp` | `False` | Save intermediate temp rasters |
| `-mp` | `--man-pwl` | `False` | Include manually digitized powerline layer |
| `-slp` | `--slope-threshold` | `None` | Slope value above which pixels are considered for masking |
| `-grt` | `--greenred-threshold` | `135` | Greenred ratio cutoff — pixels below this are masked as buildings |
| `-bt` | `--building-threshold` | `30` | Building mask height threshold |

---

## Required Data Folders

The following input paths are currently hardcoded in `Clean_CHM.py` and `Test_greenred.py`:

```
/gpfs/glad1/Theo/Data/Lidar/CHM_cleaning/
├── Canopy/Canopy.shp
├── Powerlines/
├── Manual_powerlines/
├── WorldCover/
├── Planet_tiles/
├── Building_mask_2022/
└── Slope/
```

---

## Dataset Sources

| Dataset | Source |
|---|---|
| WorldCover | [ESA WorldCover](https://github.com/ESA-WorldCover/esa-worldcover-datasets) |
| Powerlines | [OpenStreetMap via Overpass Turbo](https://overpass-turbo.eu/) |

### WorldCover Class Labels

| Value | Land Cover Type |
|---|---|
| 10 | Trees |
| 20 | Shrubland |
| 30 | Grassland |
| 40 | Cropland |
| 50 | Built-up |
| 60 | Bare areas |
| 70 | Snow / Ice |
| 80 | Permanent water bodies |
| 90 | Herbaceous wetland |
| 95 | Mangroves |
| 100 | Lichens and mosses |

---

## Environment

Designed to run on a Linux cluster in the `gdalenv` conda environment.