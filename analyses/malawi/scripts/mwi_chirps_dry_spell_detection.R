########
## Identify dry spells (less than 2 mm of rain in 14 or more consecutive days) in observational rainfall data from CHIRPS
########

## load libraries
library(tidyverse)
library(sf)
library(raster)

# load functions
source("scripts/mwi_chirps_dry_spells_functions.R")

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

chirps_projection <- projection(chirps)

# crop and mask down to MWI
chirps_cropped <- crop(x = chirps, y = extent(mwi_adm2_spatial_extent))
chirps_masked <- mask(chirps_cropped, mask = mwi_adm2)
chirps_masked
plot(chirps_masked) # visual inspection

# create list of regions
region_list <- mwi_adm2[,c('ADM2_PCODE', 'ADM2_EN', 'geometry')]

## compute precipitation totals per adm2

# loop through layers/days to compile MAX values across layers/days
nbr_layers <- nlayers(chirps_masked)
chirps_max_values <- data.frame(ID = 1:nrow(mwi_adm2))

for (i in seq_along(1:nbr_layers)) {
  
        chirps_max_values <- computeLayerStat(i, max, chirps_max_values)
        
      }

# compute per-region 14-d rolling sums
chirps_max_sums <- compute14dSum(chirps_max_values)

## identify dry spells per adm2

# list days with 14-day rolling sums of 2mm or less of rain
chirps_max_sums$dry_spell_day_bin <- ifelse(chirps_max_sums$rollsum_14d <= 2, 1, 0)

# identify beginning, end and duration of dry spells per region
dry_spells_list <- chirps_max_sums %>%
                  mutate(dry_spell_day_bin = ifelse(is.na(dry_spell_day_bin), 0, dry_spell_day_bin)) %>%  # remove NAs with 0
                  group_by(pcode, spell = cumsum(c(0, diff(dry_spell_day_bin) != 0))) %>% # groups consecutive "dry spell" days 
                  filter(dry_spell_day_bin == 1 & n() > 1) %>%
                  summarize(first_dry_spell_day = min(date), # first day on which the dry spell criterion is met (14+ days with <= 2mm of rain)
                            first_dry_day_date = first_dry_spell_day - 13, # spell started 14th day prior to first day of dry spell 
                            last_dry_day_date = max(date),
                            duration_days = as.numeric((last_dry_day_date - first_dry_day_date + 1))) %>%
                  ungroup() %>%
                  as.data.frame() %>%
                  dplyr::select(-spell)


  