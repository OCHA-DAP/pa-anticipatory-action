########
## Identify dry spells (less than 2 mm of rain in 14 or more consecutive days) in observational rainfall data from CHIRPS
########

## load libraries
library(tidyverse)
library(sf)
library(raster)


## setup
rasterOptions(maxmemory = 1e+09)

## load data 
# set file paths
shapefile_path <- shapefiles_path <- "Data/Shapefiles/mwi_adm_nso_20181016_shp/"
chirps_path <- "../../indicators/drought/Data/chirps/"

# read in country shapefiles
mwi_adm2 <- st_read(paste0(shapefiles_path, "mwi_admbnda_adm2_nso_20181016.shp"))
mwi_adm3 <- st_read(paste0(shapefiles_path, "mwi_admbnda_adm3_nso_20181016.shp"))

# explore shapefiles
summary(st_geometry_type(mwi_adm2)) # summary of geometry types
st_crs(mwi_adm2) # coordinate reference system of shapefile
st_bbox(mwi_adm2) # spatial extent 
nrow(mwi_adm2) # number of features (= spatial objects / adm2 regions) 
ncol(mwi_adm2) # number of attributes
names(mwi_adm2) # names of attributes
mwi_adm2
plot(mwi_adm2$geometry) # visual inspection

mwi_adm2_spatial_extent <- st_bbox(mwi_adm2)

# read in CHIRPS data (raster)
chirps <- brick(paste0(chirps_path, "chirps_global_daily_2010_p25.nc")) 
chirps

# explore raster files
st_crs(chirps) # coordinate reference system
st_bbox(chirps) # spatial extent
ncell(chirps) # number of cells per layer (nrow * ncol)
nrow(chirps) # number of rows in a layer
ncol(chirps) # number of columns in a layer
nlayers(chirps) # number of layers (days)
dim(chirps) # (nrow, ncol, nlayers)

# crop and mask down to MWI
chirps_cropped <- crop(x = chirps, y = extent(mwi_adm2_spatial_extent))
chirps_masked <- mask(chirps_cropped, mask = mwi_adm2)
chirps_masked
plot(chirps_masked) # visual inspection

## compute precipitation total per adm2
# extract individual layer ( = 1 day)
chirps_20100101 <- subset(chirps_masked, 1)

# compute summary statistics for every adm2
chirps_20100101.max <- extract(chirps_20100101, mwi_adm2, fun = max)
chirps_20100101.ave <- extract(chirps_20100101, mwi_adm2, fun = mean)

# add daily ave and max to vector data
mwi_adm2.data <- cbind(mwi_adm2, chirps_20100101.max) 
mwi_adm2.data <- cbind(mwi_adm2.data, chirps_20100101.ave) 

