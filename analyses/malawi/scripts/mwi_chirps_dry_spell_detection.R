########
## Project: Identify dry spells (less than 2 mm of rain in 14 or more consecutive days) in observational rainfall data from CHIRPS
########

#####
## setup
#####

# load libraries
library(tidyverse)
library(sf)
library(raster)

# load functions
source("scripts/mwi_chirps_dry_spells_functions.R")

# set options
rasterOptions(maxmemory = 1e+09)

# set directory paths
# AA_DATA_DIR is set as a variable in .Renviron or .bashprofile
data_dir <- Sys.getenv("AA_DATA_DIR")
shapefile_path <- paste0(data_dir, "/raw/malawi/Shapefiles/mwi_adm_nso_20181016_shp")
chirps_path <- paste0(data_dir, "/raw/drought/chirps")

#####
## process shapefiles
#####

# read in country shapefiles
mwi_adm2 <- st_read(paste0(shapefile_path, "/mwi_admbnda_adm2_nso_20181016.shp"))
# mwi_adm3 <- st_read(paste0(shapefile_path, "/mwi_admbnda_adm3_nso_20181016.shp"))

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

# list years and adm2 regions to be analysed
year_list <- data.frame(year = lubridate::year(seq.Date(from = as.Date("2010-01-01"), to = as.Date("2020-12-31"), by = 'year')))
year_by_adm2 <- crossing(year_list, mwi_adm2_ids$ADM2_PCODE) # create list with all year * ad2 combinations
names(year_by_adm2)[2] <- 'pcode'
year_by_adm2$year <- as.character(year_by_adm2$year)

#####
## process observational rainfall data (CHIRPS)
#####

# read in CHIRPS data (multiple multi-layer raster files) into a single stack
s2010 <- raster::stack(paste0(data_dir, "/raw/drought/chirps/chirps_global_daily_2010_p05.nc")) # each file has to be read in separately or layer names get lost
s2011 <- raster::stack(paste0(data_dir, "/raw/drought/chirps/chirps_global_daily_2011_p05.nc"))
s2012 <- raster::stack(paste0(data_dir, "/raw/drought/chirps/chirps_global_daily_2012_p05.nc"))
s2013 <- raster::stack(paste0(data_dir, "/raw/drought/chirps/chirps_global_daily_2013_p05.nc"))
s2014 <- raster::stack(paste0(data_dir, "/raw/drought/chirps/chirps_global_daily_2014_p05.nc"))
s2015 <- raster::stack(paste0(data_dir, "/raw/drought/chirps/chirps_global_daily_2015_p05.nc"))
s2016 <- raster::stack(paste0(data_dir, "/raw/drought/chirps/chirps_global_daily_2016_p05.nc"))
s2017 <- raster::stack(paste0(data_dir, "/raw/drought/chirps/chirps_global_daily_2017_p05.nc"))
s2018 <- raster::stack(paste0(data_dir, "/raw/drought/chirps/chirps_global_daily_2018_p05.nc"))
s2019 <- raster::stack(paste0(data_dir, "/raw/drought/chirps/chirps_global_daily_2019_p05.nc"))
s2020 <- raster::stack(paste0(data_dir, "/raw/drought/chirps/chirps_global_daily_2020_p05.nc"))

s2010_s2020 <- stack(s2010, s2011, s2012, s2013, s2014, s2015, s2016, s2017, s2018, s2019, s2020) # all files combined into a stack

# crop and masked area outside of MWI
s2010_s2020_cropped <- crop(x = s2010_s2020, y = extent(mwi_adm2_spatial_extent)) # crop converts to a brick - a single raster file
data <- mask(s2010_s2020_cropped, mask = mwi_adm2)
#saveRDS(data,paste0(data_dir, "/processed/malawi/dry_spells/data_20210219_r5.RDS")
#data <- readRDS(paste0(data_dir, "/processed/malawi/dry_spells/data_20210219_r5.RDS")) # 5-deg resolution
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

# create list of regions
region_list <- mwi_adm2[,c('ADM2_PCODE', 'ADM2_EN', 'geometry')]

# loop through layers/days to compile MAX values across layers/days
nbr_layers <- nlayers(data)
data_max_values <- data.frame(ID = 1:nrow(mwi_adm2))

for (i in seq_along(1:nbr_layers)) {
  
        data_max_values <- computeLayerStat(i, max, data_max_values)
        
      }
#saveRDS(data_max_values,paste0(data_dir, "/processed/malawi/dry_spells/data_max_values_20210219_r5.rds")
# data_max_values <- readRDS(paste0(data_dir, "/processed/malawi/dry_spells/data_max_values_20210219_r5.RDS"))

#####
## transform rainfall data and compute rolling sums
#####

# transpose data; create Year, Month,Day columns; label rainy season year (approximated: Oct-June to give room for early starts and late cessations)
data_max_values_long <- convertToLongFormat(data_max_values)
data_max_values_long$year <- lubridate::year(data_max_values_long$date) 
data_max_values_long$month <- lubridate::month(data_max_values_long$date) 
data_max_values_long$day <- lubridate::day(data_max_values_long$date) 
data_max_values_long$season_approx <- ifelse(data_max_values_long$month >= 10, data_max_values_long$year, ifelse(data_max_values_long$month <= 6, data_max_values_long$year - 1, 'outside rainy season')) # labels the rainy season which overlaps between two calendar years. uses first year as label.

# compute 10-day rolling sums
data_max_values_long <- data_max_values_long %>%
                          group_by(pcode) %>%
                          computeRollingSum(., window = 10) %>%
                          rename(rollsum_10d = rollsum)

# compute 14-day rolling sums
data_max_values_long <- data_max_values_long %>%
                          group_by(pcode) %>%
                          computeRollingSum(., window = 14) %>%
                          rename(rollsum_14d = rollsum)

# compute 15-day rolling sums
data_max_values_long <- data_max_values_long %>%
                          group_by(pcode) %>%
                          computeRollingSum(., window = 15) %>%
                          rename(rollsum_15d = rollsum)


#####
## identify rainy season onset/cessation/duration per year, adm2
#####

# Rainy season onset: First day of a period after 1 Nov with at least 40mm of rain over 10 days AND no 10 consecutive days with less than 2mm of total rain in the following 30 days (DCCMS 2008).
rainy_onsets <- findRainyOnset()

# Rainy season cessation: 25mm or less of rain in 15 days after 15 March (DCCMS 2008).
rainy_cessations <- findRainyCessation()

# combine onset and cessation dates
rainy_seasons <- merge(rainy_onsets, rainy_cessations, by = c('ID', 'pcode', 'season_approx'), all.x = TRUE, all.y = TRUE) # keep partial years 2009 and 2020
rainy_seasons <- merge(rainy_seasons, year_by_adm2, by.x = c('pcode', 'season_approx'), by.y = c('pcode', 'year'), all.y = T) # ensure a record is created for each adm2 for every year

# checks
sum(ifelse(rainy_seasons$cessation_date < rainy_seasons$onset_date, 1, 0), na.rm = T) # sum indicates nbr of records for which cessation date precedes onset date
nlevels(as.factor(rainy_seasons$pcode)) # number of districts in dataset
nlevels(as.factor(rainy_seasons$season_approx)) # number of seasons in dataset
nrow(rainy_seasons) == nlevels(as.factor(rainy_seasons$pcode)) * nlevels(as.factor(rainy_seasons$season_approx)) # is the number of records in rainy_seasons the multiple of number of regions and number of years?

table(rainy_seasons$pcode, rainy_seasons$season_approx) # table of available data per adm2 and year. ## TO DO: Check MW106 in 2020

# compute duration (cessation minus onset dates in days)
rainy_seasons$duration <- as.numeric(difftime(rainy_seasons$cessation_date, rainy_seasons$onset_date, units = "days"))

# create variables for exploration
rainy_seasons$onset_month <- lubridate::month(rainy_seasons$onset_date)
rainy_seasons$cessation_month <- lubridate::month(rainy_seasons$cessation_date)
  
#####
## identify dry spells that occurred during a rainy season 
#####

# label rainy days
data_max_values_long$rainy_day_bin <-  ifelse(data_max_values_long$total_prec >= 4, 1, 0) # rainy day defined as having received at least 4mm

# determine if each record is within that year's rainy season and if so, how many days into the season it is
data_max_values_long <- merge(data_max_values_long, rainy_seasons, by = c('ID', 'pcode', 'season_approx'), all.x = T)
data_max_values_long$during_rainy_season_bin <-  ifelse(data_max_values_long$date >= data_max_values_long$onset_date & data_max_values_long$date <= data_max_values_long$cessation_date, 1, 0)
data_max_values_long$nth_day_of_rainy_season <- ifelse(data_max_values_long$during_rainy_season_bin == 1, as.numeric(difftime(data_max_values_long$date, data_max_values_long$onset_date, units = "days")), NA)

# find rainy streaks within each rainy season, adm2
rainy_streaks <- data_max_values_long %>%
                  filter(during_rainy_season_bin == 1) %>% # keep days during the rainy season
                  group_by(pcode, season_approx) %>%        
                  arrange(pcode, date) %>% # sort in ascending order
                  mutate(streak_number = runlengthEncoding(rainy_day_bin)) %>% # number each group of consecutive days with/without rain per adm2 and year
                  filter(rainy_day_bin == 1) %>% # keep the rainy streaks
                  ungroup() 

# label days on which 14-day rolling sum is 2mm or less of rain as "dry_spell_day"
data_max_values_long$ds_day_bin <- ifelse(data_max_values_long$rollsum_14d <= 2, 1, 0)
#data_max_values_long$ds_day_bin <- ifelse(is.na(data_max_values_long$ds_day_bin), 0, data_max_values_long$ds_day_bin) # replace NAs with 0

# identify beginning, end and duration of dry spells per region 
dry_spells_list <- data_max_values_long %>%
                    filter(during_rainy_season_bin == 1 & !season_approx %in% c('2009', '2020')) %>% # keep streaks during rainy seasons. remove 2009 and 2020 seasons because of incomplete data
                    group_by(pcode, spell = cumsum(c(0, diff(ds_day_bin) != 0))) %>% # groups consecutive days with rolling sum <= 2mm. [c(0) is to start the array with zero]. "spell" creates IDs for streaks of rollsum_ds_bin == 0 or == 1.
                    filter(ds_day_bin == 1 & n() >= 1) %>% # keep all streaks of dates on which the 14-day rolling sum was <=2mm even if only 1 date's rolling sum met the criterion
                    summarize(dry_spell_confirmation = min(date), # first day on which the dry spell criterion is met (14+ days with <= 2mm of rain)
                              dry_spell_first_date = dry_spell_confirmation - 13, # spell started 14th day prior to first day of dry spell 
                              dry_spell_last_date = max(date),
                              dry_spell_duration = as.numeric((dry_spell_last_date - dry_spell_first_date + 1)),
                              dry_spell_total_rainfall = sum(total_prec)) %>%
                    ungroup() %>%
                    as.data.frame() %>%
                    dplyr::select(-spell)

# add region names for ease of communication
dry_spells_list <- dry_spells_list %>% 
                    left_join(mwi_adm2_ids, by = c('pcode'= 'ADM2_PCODE')) %>%
                    dplyr::select(pcode, ADM2_EN, dry_spell_confirmation, dry_spell_first_date, dry_spell_last_date, dry_spell_duration, dry_spell_total_rainfall)

# summary stats per region 
dry_spells_summary_per_region <- dry_spells_list %>%
                                    group_by(pcode, ADM2_EN) %>%
                                    summarise(nbr_dry_spells = n(),
                                              median_ds_duration = mean(dry_spell_duration),
                                              min_ds_duration = min(dry_spell_duration),
                                              max_ds_duration = max(dry_spell_duration,
                                              median_rainfall = mean(dry_spell_total_rainfall)))

dry_spells_summary_per_region <- merge(dry_spells_summary_per_region, mwi_adm2_ids, by.x = c('pcode', 'ADM2_EN'), by.y = c('ADM2_PCODE', 'ADM2_EN'), all.y = T) # ensure every region is in dataset
dry_spells_summary_per_region$nbr_dry_spells <- ifelse(is.na(dry_spells_summary_per_region$nbr_dry_spells), 0, dry_spells_summary_per_region$nbr_dry_spells) # replace NAs with 0 under nbr of dry spells


## TO DO: update below

#####
## explore rainy season patterns
#####

# check for which years adm2's don't have a start date in Nov-Dec
rainy_season_starts %>%
  group_by(pcode) %>%
  mutate(nbr_yrs_with_ND_start = sum(start_month %in% c(10:12))) %>%
  dplyr::select(pcode, nbr_yrs_with_ND_start) %>%
  unique() %>%
  data.frame()

# identify years without an ND start
rainy_season_starts %>%
  group_by(pcode) %>%
  filter(!start_month %in% c(10:12)) %>% # keep rainy seasons that started outside ND
  mutate(ND_years = paste(rainy_season, collapse = ',')) %>%
  ungroup() %>%
  dplyr::select(pcode, ADM2_EN, ND_years) %>%
  unique() %>%
  data.frame()

# check frequency of start month per adm2 ## TO DO: Verify that Blantyre and Blantyre City are separate districts (305, 315)
rainy_season_starts %>%
  dplyr::select(pcode, ADM2_EN, start_month) %>%
  mutate(start_month_name = lubridate::month(start_month, label = T, abbr = T)) %>% # names months to use a column name in next step
  dplyr::select(-start_month) %>% # remove start_month 
  group_by(pcode, ADM2_EN, start_month_name) %>%
  add_count(start_month_name) %>%
  unique() %>%
  spread(key = start_month_name, value = n) %>% # convert to wide format
  replace(is.na(.), 0) %>%
  print(n = 35)



  