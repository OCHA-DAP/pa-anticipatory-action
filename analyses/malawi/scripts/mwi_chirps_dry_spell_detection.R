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
shapefile_path <- shapefiles_path <- "Data/Shapefiles/mwi_adm_nso_20181016_shp"
chirps_path <- "../../indicators/drought/Data/chirps"

# read in country shapefiles
mwi_adm2 <- st_read(paste0(shapefiles_path, "/mwi_admbnda_adm2_nso_20181016.shp"))
mwi_adm3 <- st_read(paste0(shapefiles_path, "/mwi_admbnda_adm3_nso_20181016.shp"))

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
mwi_adm2_ids <- as.data.frame(mwi_adm2) %>% dplyr::select('ADM2_PCODE', 'ADM2_EN') 

# read in CHIRPS data (multiple multi-layer raster files) into a single stack
s2010 <- raster::stack("../../indicators/drought/Data/chirps/chirps_global_daily_2010_p25.nc") # each file has to be read in separately or layer names get lost
s2011 <- raster::stack("../../indicators/drought/Data/chirps/chirps_global_daily_2011_p25.nc")
s2012 <- raster::stack("../../indicators/drought/Data/chirps/chirps_global_daily_2012_p25.nc")
s2013 <- raster::stack("../../indicators/drought/Data/chirps/chirps_global_daily_2013_p25.nc")
s2014 <- raster::stack("../../indicators/drought/Data/chirps/chirps_global_daily_2014_p25.nc")
s2015 <- raster::stack("../../indicators/drought/Data/chirps/chirps_global_daily_2015_p25.nc")
s2016 <- raster::stack("../../indicators/drought/Data/chirps/chirps_global_daily_2016_p25.nc")
s2017 <- raster::stack("../../indicators/drought/Data/chirps/chirps_global_daily_2017_p25.nc")
s2018 <- raster::stack("../../indicators/drought/Data/chirps/chirps_global_daily_2018_p25.nc")
s2019 <- raster::stack("../../indicators/drought/Data/chirps/chirps_global_daily_2019_p25.nc")
s2020 <- raster::stack("../../indicators/drought/Data/chirps/chirps_global_daily_2020_p25.nc")

s2010_s2020 <- stack(s2010, s2011, s2012, s2013, s2014, s2015, s2016, s2017, s2018, s2019, s2020) # all files combined into a stack

# crop and masked area outside of MWI
s2010_s2020_cropped <- crop(x = s2010_s2020, y = extent(mwi_adm2_spatial_extent)) # crop converts to a brick - a single raster file
data <- mask(s2010_s2020_cropped, mask = mwi_adm2)

plot(data) # visual inspection

# explore compiled raster file ("brick")
st_crs(data) # coordinate reference system
st_bbox(data) # spatial extent
ncell(data) # number of cells per layer (nrow * ncol)
nrow(data) # number of rows in a layer
ncol(data) # number of columns in a layer
nlayers(data) # number of layers (days)
dim(data) # (nrow, ncol, nlayers)
yres(data) # y-resolution
xres(data) # x-resolution

data_projection <- projection(data) # assign CRS to variable

# create list of regions
region_list <- mwi_adm2[,c('ADM2_PCODE', 'ADM2_EN', 'geometry')]

## compute precipitation totals per adm2

# loop through layers/days to compile MAX values across layers/days
nbr_layers <- nlayers(data)
data_max_values <- data.frame(ID = 1:nrow(mwi_adm2))

for (i in seq_along(1:nbr_layers)) {
  
        data_max_values <- computeLayerStat(i, max, data_max_values)
        
      }

## identify yearly rainy season start date per region

# transpose data, create Year column
data_max_values_long <- convertToLongFormat(data_max_values)
data_max_values_long$year <- lubridate::year(data_max_values_long$date) 
data_max_values_long$month <- lubridate::month(data_max_values_long$date) 

# find rainy days (total_prec > 0) after 1 Oct every year
rainy_streaks <- data_max_values_long %>%
                  filter(month >= 10) %>%  # keep oct-nov-dec data
                  mutate(rainy_day_bin = ifelse(total_prec > 0, 1, 0)) %>%
                  group_by(year, pcode) %>%        
                  arrange(pcode, date) %>%
                  mutate(streak_number = runlengthEncoding(rainy_day_bin)) %>% # number each group of consecutive days with/without rain per adm2 and year
                  filter(rainy_day_bin == 1) %>% # keep the rainy streaks
                  ungroup 

# find earliest streak with 7+ days per adm2 and year
rainy_season_starts <- rainy_streaks %>%
                        arrange(pcode, year, date) %>%
                        group_by(year, pcode, streak_number) %>%
                        mutate(streak_length = n()) %>% # count nbr of days in streaks
                        filter(streak_length >= 7) %>% # keep streaks of at least 7 days
                        ungroup(streak_number) %>% # remove streak_number from grouping
                        slice(which.min(date)) %>% # select earliest per year and pcode
                        ungroup %>%  
                        dplyr::select(year, pcode, date, streak_length) %>% # remove and reorder columns
                        rename(earliest_streak_start_date = date) %>%
                        data.frame() %>%
                        left_join(mwi_adm2_ids, by = c('pcode' = 'ADM2_PCODE')) 
                                                
# check that there is a start date per adm2 and year
nrow(rainy_season_starts) == (n_distinct(mwi_adm2_ids$ADM2_PCODE)*11)

# check for which years adm2's don't have a start date in Oct-Dec
rainy_season_starts %>%
        group_by(pcode) %>%
        mutate(nbr_yrs_with_OND_start = n_distinct(year)) %>%
        dplyr::select(pcode, nbr_yrs_with_OND_start) %>%
        unique() %>%
        data.frame()

# identify years without an OND start
rainy_season_starts %>%
  group_by(pcode) %>%
  mutate(OND_years = paste(year, collapse = ',')) %>%
  ungroup() %>%
  dplyr::select(pcode, ADM2_EN, OND_years) %>%
  unique() %>%
  data.frame()

## identify dry spells per adm2

# compute per-region 14-d rolling sums
data_max_sums <- compute14dSum(data_max_values)

# list days on which 14-day rolling sum is 2mm or less of rain
data_max_sums$rollsum_ds_bin <- ifelse(data_max_sums$rollsum_14d <= 2, 1, 0)

# identify beginning, end and duration of dry spells per region
dry_spells_list <- data_max_sums %>%
                      mutate(rollsum_ds_bin = ifelse(is.na(rollsum_ds_bin), 0, rollsum_ds_bin)) %>%  # replace NAs with 0
                      group_by(pcode, spell = cumsum(c(0, diff(rollsum_ds_bin) != 0))) %>% # groups consecutive days with rolling sum <= 2mm. [c(0) is to start the array with zero]
                      filter(rollsum_ds_bin == 1 & n() > 1) %>%
                      summarize(dry_spell_confirmation = min(date), # first day on which the dry spell criterion is met (14+ days with <= 2mm of rain)
                                dry_spell_first_date = dry_spell_confirmation - 13, # spell started 14th day prior to first day of dry spell 
                                dry_spell_last_date = max(date),
                                duration_days = as.numeric((dry_spell_last_date - dry_spell_first_date + 1))) %>%
                      ungroup() %>%
                      as.data.frame() %>%
                      dplyr::select(-spell)

# add region names for ease of communication
dry_spells_list <- dry_spells_list %>% 
                      left_join(mwi_adm2_ids, by = c('pcode'= 'ADM2_PCODE')) %>%
                      dplyr::select(pcode, ADM2_EN, dry_spell_confirmation, dry_spell_first_date, dry_spell_last_date, duration_days)

# summary stats per region ## FIX ME: no CHIRPS data for Likoma, Mwanza?
dry_spells_summary_per_region <- dry_spells_list %>%
                                    group_by(pcode, ADM2_EN) %>%
                                    summarise(nbr_dry_spells = n(),
                                              median_duration = mean(duration_days),
                                              min_duration = min(duration_days),
                                              max_duration = max(duration_days))



  