########
## Project: Identify dry spells in observational rainfall data from CHIRPS at ADM3 level
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
mwi_adm3 <- st_read(paste0(shapefile_path, "/mwi_admbnda_adm3_nso_20181016.shp"))

# explore shapefiles
summary(st_geometry_type(mwi_adm3)) # summary of geometry types
st_crs(mwi_adm3) # coordinate reference system of shapefile
st_bbox(mwi_adm3) # spatial extent 
nrow(mwi_adm3) # number of features (= spatial objects / adm3 regions) 
ncol(mwi_adm3) # number of attributes
names(mwi_adm3) # names of attributes
mwi_adm3
plot(mwi_adm3$geometry) # visual inspection

mwi_adm3_spatial_extent <- st_bbox(mwi_adm3)
mwi_adm3_ids <- as.data.frame(mwi_adm3) %>% dplyr::select('ADM3_PCODE', 'ADM3_EN')
mwi_adm3_full_ids <- as.data.frame(mwi_adm3) %>% dplyr::select('ADM3_PCODE', 'ADM3_EN', 'ADM2_PCODE', 'ADM2_EN', 'ADM1_PCODE', 'ADM1_EN')

# list years and adm3 regions to be analysed
year_list <- data.frame(year = lubridate::year(seq.Date(from = as.Date("2000-01-01"), to = as.Date("2020-12-31"), by = 'year')))
year_by_adm3 <- crossing(year_list, mwi_adm3_ids$ADM3_PCODE) # create list with all year * ad2 combinations
names(year_by_adm3)[2] <- 'pcode'
year_by_adm3$year <- as.character(year_by_adm3$year)

#####
## process observational rainfall data (CHIRPS)
#####

# read in CHIRPS data (multiple multi-layer raster files) into a single stack
s2000 <- raster::stack(paste0(data_dir, "/raw/drought/chirps/chirps_global_daily_2000_p05.nc")) # each file has to be read in separately or layer names get lost
s2001 <- raster::stack(paste0(data_dir, "/raw/drought/chirps/chirps_global_daily_2001_p05.nc"))
s2002 <- raster::stack(paste0(data_dir, "/raw/drought/chirps/chirps_global_daily_2002_p05.nc"))
s2003 <- raster::stack(paste0(data_dir, "/raw/drought/chirps/chirps_global_daily_2003_p05.nc"))
s2004 <- raster::stack(paste0(data_dir, "/raw/drought/chirps/chirps_global_daily_2004_p05.nc"))
s2005 <- raster::stack(paste0(data_dir, "/raw/drought/chirps/chirps_global_daily_2005_p05.nc"))
s2006 <- raster::stack(paste0(data_dir, "/raw/drought/chirps/chirps_global_daily_2006_p05.nc"))
s2007 <- raster::stack(paste0(data_dir, "/raw/drought/chirps/chirps_global_daily_2007_p05.nc"))
s2008 <- raster::stack(paste0(data_dir, "/raw/drought/chirps/chirps_global_daily_2008_p05.nc"))
s2009 <- raster::stack(paste0(data_dir, "/raw/drought/chirps/chirps_global_daily_2009_p05.nc"))
s2010 <- raster::stack(paste0(data_dir, "/raw/drought/chirps/chirps_global_daily_2010_p05.nc")) 
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

s2000_s2020 <- stack(s2000, s2001, s2002, s2003, s2004, s2005, s2006, s2007, s2008, s2009, s2010, s2011, s2012, s2013, s2014, s2015, s2016, s2017, s2018, s2019, s2020) # all files combined into a stack

# crop to MWI area
s2000_s2020_cropped <- crop(x = s2000_s2020, y = extent(mwi_adm3_spatial_extent)) # crop converts to a brick - a single raster file

# save or read as a raster file. Identical to adm2, adm1. (do not save as RDS or tif as it changes the variable names)
#writeRaster(s2000_s2020_cropped, filename = paste0(data_dir, '/processed/malawi/dry_spells/s2000_s2020_cropped.grd'), bandorrder='BIL', overwrite=TRUE)
#s2000_s2020_cropped <- brick(paste0(data_dir, "/processed/malawi/dry_spells/s2000_s2020_cropped.grd")) # read in raster (brick) file
 
data_all <- s2000_s2020_cropped
nbr_layers <- nlayers(data_all)

# plot(data_all) # visual inspection

# explore compiled raster file ("brick")
st_crs(data_all) # coordinate reference system
st_bbox(data_all) # spatial extent
ncell(data_all) # number of cells per layer (nrow * ncol)
nrow(data_all) # number of rows in a layer
ncol(data_all) # number of columns in a layer
nlayers(data_all) # number of layers (days)
dim(data_all) # (nrow, ncol, nlayers)
yres(data_all) # y-resolution
xres(data_all) # x-resolution

# create list of adm3s
adm3_list <- mwi_adm3[,c('ADM3_PCODE', 'ADM3_EN', 'geometry')]

# get mean values per polygon

## do not use the loop approach: takes two days to run and fails. using extract's nl is much more efficient esp for this number of polygons
# loop through layers/days to compile MEAN values across layers/days
# data_mean_values <- data.frame(ID = 1:nrow(mwi_adm3))
# 
# for (i in seq_along(1:nbr_layers)) {
#   
#   data_mean_values <- computeLayerStat_adm3(i, mean, data_mean_values)
#   
# }

# extract all values per polygon
all_years_values_adm3s <- raster::extract(s2000_s2020_cropped, mwi_adm3, cellnumbers = T, df = T, nl = nbr_layers) 

# compute mean per polygon
data_mean_values <- all_years_values_adm3s %>%
                          group_by(ID) %>%
                          summarise(across(2:nbr_layers+1, mean))

# saveRDS(data_mean_values, paste0(data_dir, "/processed/malawi/dry_spells/data_mean_values_2000_2020_r5_adm3.RDS"))
#data_mean_values <- readRDS(paste0(data_dir, "/processed/malawi/dry_spells/data_mean_values_2000_2020_r5_adm3.RDS")) 

data <- data_mean_values

#####
## transform rainfall data and compute rolling sums
#####

# transpose data; create Year, Month, Day columns; label rainy season year (approximated: Oct-June to give room for early starts and late cessations)
data_long <- convertToLongFormatADM3(data)
data_long$year <- lubridate::year(data_long$date) 
data_long$month <- lubridate::month(data_long$date) 
data_long$day <- lubridate::day(data_long$date) 
data_long$season_approx <- ifelse(data_long$month >= 10, data_long$year, ifelse(data_long$month <= 7, data_long$year - 1, 'outside rainy season')) # labels the rainy season which overlaps between two calendar years. uses first year as label.

# compute 10-day rolling sums
data_long <- data_long %>%
                          group_by(pcode) %>%
                          computeRollingSum(., window = 10) %>%
                          rename(rollsum_10d = rollsum)

# compute 14-day rolling sums
data_long <- data_long %>%
                          group_by(pcode) %>%
                          computeRollingSum(., window = 14) %>%
                          rename(rollsum_14d = rollsum)

# compute 15-day rolling sums
data_long <- data_long %>%
                          group_by(pcode) %>%
                          computeRollingSum(., window = 15) %>%
                          rename(rollsum_15d = rollsum)

# compute 15-day backwards rolling sums 
data_long <- data_long %>%
                          group_by(pcode) %>%
                          computeBackRollingSum(., window = 15) %>%
                          rename(rollsum_15d_back = rollsum)

# label rainy days
data_long$rainy_day_bin <-  ifelse(data_long$total_prec >= 4, 1, 0) # rainy day defined as having received at least 4mm
data_long$rainy_day_bin_2mm <-  ifelse(data_long$total_prec >= 2, 1, 0) # rainy day defined as having received at least 2mm


#####
## identify rainy season onset/cessation/duration per year, adm3
#####

# Rainy season onset: First day of a period after 1 Nov with at least 40mm of rain over 10 days AND no 10 consecutive days with less than 2mm of total rain in the following 30 days (DCCMS 2008).
rainy_onsets <- findRainyOnset()

# Rainy season cessation: 25mm or less of rain in 15 days after 15 March (DCCMS 2008). ## TO DO: Take last rainy day before the 15-day period as cessation date
rainy_cessations <- findRainyCessation()

# combine onset and cessation dates
rainy_seasons <- merge(rainy_onsets, rainy_cessations, by = c('ID', 'pcode', 'season_approx'), all.x = TRUE, all.y = TRUE) # keep partial years 2000 and 2020
rainy_seasons <- merge(rainy_seasons, year_by_adm3, by.x = c('pcode', 'season_approx'), by.y = c('pcode', 'year'), all.x = T, all.y = T) # ensure a record is created for each adm3 for every year

# checks
sum(ifelse(rainy_seasons$cessation_date < rainy_seasons$onset_date, 1, 0), na.rm = T) # sum indicates nbr of records for which cessation date precedes onset date
nlevels(as.factor(rainy_seasons$pcode)) # number of districts in dataset
nlevels(as.factor(rainy_seasons$season_approx)) # number of seasons in dataset (including partial seasons)
nrow(rainy_seasons) == nlevels(as.factor(rainy_seasons$pcode)) * nlevels(as.factor(rainy_seasons$season_approx)) # is the number of records in rainy_seasons the multiple of number of regions and number of years?

table(rainy_seasons$pcode, rainy_seasons$season_approx) # table of available data per adm3 and year. 

# compute duration (cessation minus onset dates in days)
rainy_seasons$rainy_season_duration <- as.numeric(difftime(rainy_seasons$cessation_date, rainy_seasons$onset_date, units = "days") + 1) # + 1 to include the last day of the season

# create variables for exploration
rainy_seasons$onset_month <- lubridate::month(rainy_seasons$onset_date)
rainy_seasons$cessation_month <- lubridate::month(rainy_seasons$cessation_date)

# compute total rainfall during rainy season
rainfall_during_rainy_seasons_list <- sqldf::sqldf("select m.*,
                                                r.onset_date,
                                                r.cessation_date,
                                                r.rainy_season_duration,
                                                r.onset_month,
                                                r.cessation_month
                                               from data_long m
                                               inner join rainy_seasons r 
                                               on m.pcode = r.pcode 
                                                 and m.date between r.onset_date and r.cessation_date") # keep all records during a rainy season. Will exclude 1999 and 2020 rainy seasons because don't have onset/cessation dates for them

rainfall_during_rainy_seasons_stats <- rainfall_during_rainy_seasons_list %>%
                                        group_by(pcode, season_approx) %>%
                                        summarise(n_days = n(),
                                                  rainy_season_rainfall = round(sum(total_prec), 1))

rainy_seasons_detail <- rainy_seasons %>%
                                  left_join(rainfall_during_rainy_seasons_stats, by = c('pcode' = 'pcode', 'season_approx' = 'season_approx', 'rainy_season_duration' = 'n_days')) %>%
                                  left_join(mwi_adm3_ids, by = c('pcode'= 'ADM3_PCODE')) %>%
                                  dplyr::select(ID, pcode, ADM3_EN, season_approx, onset_date, onset_month, cessation_date, cessation_month, rainy_season_duration, rainy_season_rainfall)

nrow(rainy_seasons) == nrow(rainy_seasons_detail) # check that all records were kept
nrow(rainy_seasons_detail) / n_distinct(mwi_adm3_ids$ADM3_PCODE) == 22 # confirms there is a record for every year and every adm3

rainy_seasons_detail <- rainy_seasons_detail %>% 
                          left_join(mwi_adm3[, c('ADM3_PCODE', 'ADM2_PCODE', 'ADM2_EN', 'ADM1_PCODE', 'ADM1_EN')], by = c('pcode'= 'ADM3_PCODE'))

# save results without geometry column
#write.csv(rainy_seasons_detail[, -15], file = paste0(data_dir, "/processed/malawi/dry_spells/rainy_seasons_detail_2000_2020_ADM3.csv"), row.names = FALSE)

#####
## explore rainy season patterns
#####

# onset, cessation, duration of rainy seasons by TA Note: some pcodes share an EN name
prop.table(table(rainy_seasons_detail$pcode, rainy_seasons_detail$onset_month), 1)
prop.table(table(rainy_seasons_detail$pcode, rainy_seasons_detail$cessation_month), 1)

rainy_seasons_summary_per_ta <- rainy_seasons_detail %>%
                                      mutate(nov1 = as.Date(paste0(season_approx, '-11-01'), format = "%Y-%m-%d"), # 1 nov before the onset of the season
                                             onset_days_since_nov1 = as.numeric(difftime(onset_date, nov1, units = "days")), # count of days since 1 nov
                                             cessation_days_since_nov1 = as.numeric(difftime(cessation_date, nov1, units = "days")), # count of days since 1 nov
                                             rainy_season_at_least_125d = ifelse(rainy_season_duration >= 125, 1, 0)) %>% # 125 days is length of maize growing season
                                      group_by(ADM3_EN) %>% # group by TA
                                      summarise(min_rainy_season_onset_postnov1 = min(onset_days_since_nov1, na.rm = T), # na.rm to remove incomplete seasons
                                                max_rainy_season_onset_post1nov = max(onset_days_since_nov1, na.rm = T),
                                                mean_rainy_season_onset_post1nov = mean(onset_days_since_nov1, na.rm = T), # average nbr of days since 1 Nov
                                                min_rainy_season_cessation_postnov1 = min(cessation_days_since_nov1, na.rm = T), 
                                                max_rainy_season_cessation_postnov1 = max(cessation_days_since_nov1, na.rm = T),
                                                mean_rainy_season_cessation_post1nov = mean(cessation_days_since_nov1, na.rm = T), # average nbr of days since 1 Nov
                                                min_rainy_season_duration = min(rainy_season_duration, na.rm = T), 
                                                max_rainy_season_duration = max(rainy_season_duration, na.rm = T),
                                                mean_rainy_season_duration = mean(rainy_season_duration, na.rm = T),
                                                nbr_125d_seasons = sum(rainy_season_at_least_125d, na.rm = T),
                                                min_rainy_season_rainfall = min(rainy_season_rainfall, na.rm = T), 
                                                max_rainy_season_rainfall = max(rainy_season_rainfall, na.rm = T),
                                                mean_rainy_season_rainfall = mean(rainy_season_rainfall, na.rm = T)) %>%
                                      ungroup() %>%
                                      unique() %>%
                                      data.frame()

rainy_seasons_summary_per_ta

#####
## identify dry spells that occurred during a rainy season 
#####

# determine if each record is within that year's rainy season and if so, how many days into the season it is
data_long <- merge(data_long, rainy_seasons, by = c('ID', 'pcode', 'season_approx'), all.x = T)
data_long$during_rainy_season_bin <-  ifelse(data_long$date >= data_long$onset_date & data_long$date <= data_long$cessation_date, 1, 0)
data_long$nth_day_of_rainy_season <- ifelse(data_long$during_rainy_season_bin == 1, as.numeric(difftime(data_long$date, data_long$onset_date, units = "days") + 1), NA) # +1 so first day of rainy season is labelled "one"

# label days on which 14-day rolling sum is 2mm or less of rain as "dry_spell_day"
data_long$rollsum_14d_less_than_2_bin <- ifelse(data_long$rollsum_14d <= 2, 1, 0) # NOTE: this does not label all days that have less than 2mm because those in the first 13 days don't get flagged

# identify beginning, end and duration of dry spells per adm3 (total <= 2mm)
dry_spells_confirmation_dates <- data_long %>%
                                  group_by(pcode) %>%        
                                  arrange(date) %>% # sort date in ascending order
                                  mutate(streak_number = runlengthEncoding(rollsum_14d_less_than_2_bin)) %>% # assign numbers to streaks of days that meet/don't meet the dry spell criterion (criterion: 14d rolling sum <= 2mm)
                                  filter(rollsum_14d_less_than_2_bin == 1) %>% # keep streaks that meet the dry spell criterion
                                  ungroup()

dry_spells_list <- dry_spells_confirmation_dates %>%
                        group_by(pcode, season_approx, streak_number) %>% # for each dry spell of every adm3 and rainy season
                        summarize(dry_spell_confirmation = min(date), # first day on which the dry spell criterion is met (= 14d rolling sum of <= 2mm of rain)
                                  dry_spell_first_date = dry_spell_confirmation - 13, # spell started 14th day prior to confirmation day (= first day with a 14d rolling sum below 2mm)
                                  dry_spell_last_date = max(date),
                                  dry_spell_duration = as.numeric(dry_spell_last_date - dry_spell_first_date + 1)) %>% # do not compute total rainfall because 13 days before confirmation date not included in streak
                        ungroup() %>%
                        as.data.frame() %>%
                        dplyr::select(-streak_number)

rainfall_during_dry_spells <- sqldf::sqldf("select m.*, 
                                             l.dry_spell_confirmation,
                                             l.dry_spell_first_date,
                                             l.dry_spell_last_date
                                           from data_long m
                                           inner join dry_spells_list l 
                                           on m.pcode = l.pcode 
                                            and m.date between l.dry_spell_first_date and l.dry_spell_last_date") # keep all records during a dry spell

rainfall_during_dry_spells_stats <- rainfall_during_dry_spells %>%
                                      group_by(pcode, dry_spell_confirmation) %>%
                                      summarise(n_days = n(),
                                                dry_spell_rainfall = round(sum(total_prec), 1))

dry_spells_details <- dry_spells_list %>%
                        left_join(rainfall_during_dry_spells_stats, by = c('pcode' = 'pcode', 'dry_spell_confirmation' = 'dry_spell_confirmation', 'dry_spell_duration' = 'n_days'))

nrow(dry_spells_list) == nrow(dry_spells_details) # check that all records were kept

# identify dry spells during rainy seasons
dry_spells_during_rainy_season_list <- dry_spells_details %>%
                                          left_join(rainy_seasons[, c('pcode', 'season_approx', 'onset_date', 'cessation_date')], by = c('pcode', 'season_approx'), all.x = T, all.y = T) %>% # add rainy onset and cessation dates
                                          mutate(confirmation_date_during_rainy_season = ifelse(dry_spell_confirmation >= onset_date & dry_spell_confirmation <= cessation_date, 1, 0)) %>% # identifies dry spells that reached 14-d rolling sum during rainy season 
                                          filter(confirmation_date_during_rainy_season == 1) %>% # only keep dry spells that were confirmed during rainy season even if started before onset or ended after cessation
                                          dplyr::select(pcode, season_approx, dry_spell_first_date, dry_spell_last_date, dry_spell_duration, dry_spell_rainfall)

# add region names for ease of communication
dry_spells_during_rainy_season_list <- dry_spells_during_rainy_season_list %>% 
                                          left_join(mwi_adm3, by = c('pcode'= 'ADM3_PCODE')) %>%
                                          dplyr::select(pcode, ADM3_EN, ADM2_EN, ADM1_EN, season_approx, dry_spell_first_date, dry_spell_last_date, dry_spell_duration, dry_spell_rainfall)

#write.csv(dry_spells_during_rainy_season_list, file = paste0(data_dir, "/processed/malawi/dry_spells/dry_spells_during_rainy_season_list_2000_2020_adm3.csv"), row.names = FALSE)


# save full list of dry spells
full_list_dry_spells <- dry_spells_details %>%
                          left_join(rainy_seasons[, c('pcode', 'season_approx', 'onset_date', 'cessation_date')], by = c('pcode', 'season_approx'), all.x = T, all.y = T) %>% # add rainy onset and cessation dates
                          mutate(confirmation_date_during_rainy_season = ifelse(dry_spell_confirmation >= onset_date & dry_spell_confirmation <= cessation_date, 1, 0)) %>% # identifies dry spells that reached 14-d rolling sum during rainy season 
                          dplyr::select(pcode, season_approx, dry_spell_first_date, dry_spell_last_date, dry_spell_duration, dry_spell_rainfall)
#write.csv(full_list_dry_spells, file = paste0(data_dir, "/processed/malawi/dry_spells/full_list_dry_spells_adm3.csv"), row.names = FALSE)

# summary stats per TA 
rainy_season_dry_spells_summary_per_ta <- dry_spells_during_rainy_season_list %>% 
                                                group_by(pcode) %>%
                                                summarise(nbr_dry_spells = n(),
                                                          mean_ds_duration = round(mean(dry_spell_duration),1),
                                                          min_ds_duration = min(dry_spell_duration),
                                                          max_ds_duration = max(dry_spell_duration)
                                                          ) %>%
                                                ungroup() %>%
                                                as.data.frame()

rainy_season_dry_spells_summary_per_ta <- merge(rainy_season_dry_spells_summary_per_ta, unique(mwi_adm3_full_ids[, c('ADM3_EN', 'ADM3_PCODE' ,'ADM2_EN', 'ADM2_PCODE', 'ADM1_EN')]), by.x = c('pcode'), by.y = c('ADM3_PCODE'), all.y = T) # ensure every region is in dataset
rainy_season_dry_spells_summary_per_ta$nbr_dry_spells <- ifelse(is.na(rainy_season_dry_spells_summary_per_ta$nbr_dry_spells), 0, rainy_season_dry_spells_summary_per_ta$nbr_dry_spells) # replace NAs with 0 under nbr of dry spells

rainy_season_dry_spells_summary_per_ta

#####
## explore rainy-season dry spells patterns
#####

# how frequently have rainy-season dry spells occurred over the years and across regions/districts?
summary(rainy_season_dry_spells_summary_per_ta$nbr_dry_spells)
prop.table(table(rainy_season_dry_spells_summary_per_ta$nbr_dry_spells))

# how many and which TAs have not had a rainy-season dry spell?
rainy_season_dry_spells_summary_per_ta %>% filter(nbr_dry_spells == 0) %>% summarise(n = n_distinct(pcode)) 
rainy_season_dry_spells_summary_per_ta %>% filter(nbr_dry_spells == 0) %>% dplyr::select(pcode) %>% unique()

# how many and which TAs have had a rainy-season dry spells?
rainy_season_dry_spells_summary_per_ta %>% filter(nbr_dry_spells > 0) %>% summarise(n = n_distinct(pcode)) 
rainy_season_dry_spells_summary_per_ta %>% filter(nbr_dry_spells > 0) %>% dplyr::select(pcode) %>% unique()

# when did the dry spells start in each TA?
prop.table(table(lubridate::month(dry_spells_during_rainy_season_list$dry_spell_first_date)))
prop.table(table(dry_spells_during_rainy_season_list$pcode, lubridate::month(dry_spells_during_rainy_season_list$dry_spell_first_date)), 1)

# when did the dry spells end in each TA?
prop.table(table(lubridate::month(dry_spells_during_rainy_season_list$dry_spell_last_date)))
prop.table(table(dry_spells_during_rainy_season_list$pcode, lubridate::month(dry_spells_during_rainy_season_list$dry_spell_last_date)), 1)



  