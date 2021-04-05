########
## Project: Identify dry spells (less than 2 mm of rain in 14 or more consecutive days) in observational rainfall data from CHIRPS
## PER PIXEL COVERAGE
########

#####
## setup
#####

# load libraries
library(tidyverse)
library(sf)
library(raster)
library(data.table)

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
mwi_adm1 <- st_read(paste0(shapefile_path, "/mwi_admbnda_adm1_nso_20181016.shp"))

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
year_list <- data.frame(year = lubridate::year(seq.Date(from = as.Date("2000-01-01"), to = as.Date("2020-12-31"), by = 'year')))
year_by_adm2 <- crossing(year_list, mwi_adm2_ids$ADM2_PCODE) # create list with all year * ad2 combinations
names(year_by_adm2)[2] <- 'pcode'
year_by_adm2$year <- as.character(year_by_adm2$year)

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

# crop area outside of MWI
s2000_s2020_cropped <- crop(x = s2000_s2020, y = extent(mwi_adm2_spatial_extent)) # crop automatically converts to a brick - a single raster file
nbr_layers <- nlayers(s2000_s2020_cropped)

# save as a raster file not RDS
#writeRaster(s2000_s2020_cropped, filename = paste0(data_dir, '/processed/malawi/dry_spells/s2000_s2020_cropped.tif'), format="GTiff", overwrite=TRUE, options=c("INTERLEAVE=BAND","COMPRESS=LZW"))
#s2000_s2020_cropped <- brick(paste0(data_dir, "/processed/malawi/dry_spells/s2000_s2020_cropped.tif")) # read in saved raster file

# extract every cell value over 20 years (Note: not exploding multipart polygons)
all_years_cell_values <- raster::extract(s2000_s2020_cropped, mwi_adm2, cellnumbers = T, df = T, nl = nbr_layers) # return a df with polygon id, cell id, and values per cell for every day of 2000-2020 (1 day/layer at a time)

# add pcode for each region
mwi_adm2_ids$ID <- seq_along(1:n_distinct(mwi_adm2_ids$ADM2_PCODE)) # creates IDs based on order, which is used by extract to assign its IDs
all_years_cell_values <- all_years_cell_values %>%
                          left_join(mwi_adm2_ids, by = 'ID') %>%
                          relocate(ADM2_PCODE, .after = ID) %>% # move the identification columns to the front of the dated values
                          relocate(ADM2_EN, .after = ADM2_PCODE)

# save the dataframe
#saveRDS(all_years_cell_values, paste0(data_dir, "/processed/malawi/dry_spells/all_years_cell_values_adm2.RDS"))
#all_years_cell_values <- readRDS(paste0(data_dir, "/processed/malawi/dry_spells/all_years_cell_values_adm2.RDS"))

#####
## reformat
#####
#data <- readRDS(paste0(data_dir, "/processed/malawi/dry_spells/all_years_cell_values_adm2.RDS"))
data <- all_years_cell_values
length(data$cell) == n_distinct(data$cell) # check that IDs are unique
cell_ids <- data.frame(unique(data$cell))
names(cell_ids)[1] <- 'cell'

cell_adms <- all_years_cell_values %>%
              dplyr::select(cell, ADM2_EN, ADM2_PCODE) %>%
              mutate(cell = as.character(cell)) %>%
              mutate(region_code = substr(ADM2_PCODE, 3, 3)) %>% 
              mutate(region = ifelse(region_code == 3, "Southern", ifelse(region_code == 2, "Central", "Northern"))) %>%
              dplyr::select(-region_code)

year_by_cell <- crossing(year_list, cell_ids$cell) # create list with all year * cell combinations
names(year_by_cell)[2] <- 'cell'
year_by_cell$year <- as.character(year_by_cell$year)
year_by_cell$cell <- as.character(year_by_cell$cell)
  
# transpose data
data_long <- gather(data, date, total_prec, 5:(nbr_layers + 4)) # convert wide to long to get dates as rows ('cell' = raster cell number)
data_long$total_prec[is.na(data_long$total_prec)] <- 0 # assign "zero" values to NA in total_prec
data_long$date <- as.Date(data_long$date, format = "X%Y.%m.%d") # reformat date

# create Year, Month, Day columns; label rainy season year (approximated: Oct-June to give room for early starts and late cessations)
data_long$year <- lubridate::year(data_long$date) 
data_long$month <- lubridate::month(data_long$date) 
data_long$day <- lubridate::day(data_long$date) 
data_long$season_approx <- ifelse(data_long$month >= 10, data_long$year, ifelse(data_long$month <= 7, data_long$year - 1, 'outside rainy season')) # labels the rainy season which overlaps between two calendar years. uses first year as label.

#####
## compute rolling sums per cell/pixel
#####

# compute 10-day rolling sums
data_long <- data_long %>%
                          computeRollingSumPerPixel(., window = 10) %>%
                          rename(rollsum_10d = rollsum)

# compute 14-day rolling sums
data_long <- data_long %>%
                          computeRollingSumPerPixel(., window = 14) %>%
                          rename(rollsum_14d = rollsum)

# compute 15-day rolling sums
data_long <- data_long %>%
                          computeRollingSumPerPixel(., window = 15) %>%
                          rename(rollsum_15d = rollsum)

# label rainy days
data_long$rainy_day_bin <-  ifelse(data_long$total_prec >= 4, 1, 0) # rainy day defined as having received at least 4mm
data_long$rainy_day_bin_2mm <-  ifelse(data_long$total_prec >= 2, 1, 0) # rainy day defined as having received at least 2mm
data_long$rainy_day_bin_8mm <-  ifelse(data_long$total_prec >= 8, 1, 0) # rainy day defined as having received at least 8mm (DCCMS' definition in 2021)

head(data_long)

#####
## identify rainy season onset/cessation/duration per year, cell
#####

# Rainy season onset: First day of a period after 1 Nov with at least 40mm of rain over 10 days AND no 10 consecutive days with less than 2mm of total rain in the following 30 days (DCCMS 2008).
rainy_onsets <- findRainyOnsetPerPixel()

# Rainy season cessation: 25mm or less of rain in 15 days after 15 March (DCCMS 2008). ## TO DO: Take last rainy day before the 15-day period as cessation date
rainy_cessations <- findRainyCessationPerPixel()

# combine onset and cessation dates
rainy_seasons <- merge(rainy_onsets, rainy_cessations, by = c('cell', 'season_approx'), all.x = TRUE, all.y = TRUE) # keep partial years 2000 and 2020
rainy_seasons <- merge(rainy_seasons, year_by_cell, by.x = c('cell', 'season_approx'), by.y = c('cell', 'year'), all.x = T, all.y = T) # ensure a record is created for each cell for every year

# checks
sum(ifelse(rainy_seasons$cessation_date < rainy_seasons$onset_date, 1, 0), na.rm = T) # sum indicates nbr of records for which cessation date precedes onset date
nlevels(as.factor(rainy_seasons$cell)) # number of districts in dataset
nlevels(as.factor(rainy_seasons$season_approx)) # number of seasons in dataset (including partial seasons)
nrow(rainy_seasons) == nlevels(as.factor(rainy_seasons$cell)) * nlevels(as.factor(rainy_seasons$season_approx)) # is the number of records in rainy_seasons the multiple of number of regions and number of years?

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
                                                 on m.cell = r.cell 
                                                 and m.date between r.onset_date and r.cessation_date") # keep all records during a rainy season. Will exclude 1999 and 2020 rainy seasons because don't have onset/cessation dates for them

rainfall_during_rainy_seasons_stats <- rainfall_during_rainy_seasons_list %>%
                                        group_by(cell, season_approx) %>%
                                        summarise(n_days = n(),
                                                  rainy_season_rainfall = round(sum(total_prec), 1)) %>%
                                        mutate(cell = as.character(cell))

rainy_seasons_detail <- rainy_seasons %>%
                                  left_join(rainfall_during_rainy_seasons_stats, by = c('cell' = 'cell', 'season_approx' = 'season_approx', 'rainy_season_duration' = 'n_days')) %>%
                                  left_join(cell_adms, by = 'cell')

nrow(rainy_seasons) == nrow(rainy_seasons_detail) # check that all records were kept
nrow(rainy_seasons_detail) / 22 == n_distinct(data$cell) # confirm there is a record for every year and every cell

# save results
#write.csv(rainy_seasons_detail, file = paste0(data_dir, "/processed/malawi/dry_spells/rainy_seasons_detail_2000_2020_per_pixel.csv"), row.names = FALSE)

#####
## identify dry spells that occurred during a rainy season 
#####

# determine if each record is within that year's rainy season and if so, how many days into the season it is
combined <- merge(data_long, rainy_seasons, by = c('cell', 'season_approx'), all.x = T)
combined$during_rainy_season_bin <-  ifelse(combined$date >= combined$onset_date & combined$date <= combined$cessation_date, 1, 0)
combined$nth_day_of_rainy_season <- ifelse(combined$during_rainy_season_bin == 1, as.numeric(difftime(combined$date, combined$onset_date, units = "days") + 1), NA) # +1 so first day of rainy season is labelled "one"

# label days on which 14-day rolling sum is 2mm or less of rain as "dry_spell_day"
combined$rollsum_14d_less_than_2_bin <- ifelse(combined$rollsum_14d <= 2, 1, 0) # NOTE: this does not label all days that have less than 2mm because those in the first 13 days don't get flagged

# identify beginning, end and duration of dry spells per cell (total <= 2mm)
dry_spells_confirmation_dates <- combined %>%
                                  group_by(cell) %>%        
                                  arrange(date) %>% # sort date in ascending order
                                  mutate(streak_number = runlengthEncoding(rollsum_14d_less_than_2_bin)) %>% # assign numbers to streaks of days that meet/don't meet the dry spell criterion (criterion: 14d rolling sum <= 2mm)
                                  filter(rollsum_14d_less_than_2_bin == 1) %>% # keep streaks that meet the dry spell criterion
                                  ungroup()

dry_spells_list <- dry_spells_confirmation_dates %>%
                        group_by(cell, season_approx, streak_number) %>% # for each dry spell of every adm2 and rainy season
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
                                           on m.cell = l.cell 
                                            and m.date between l.dry_spell_first_date and l.dry_spell_last_date") # keep all records during a dry spell

rainfall_during_dry_spells_stats <- rainfall_during_dry_spells %>%
                                      group_by(cell, dry_spell_confirmation) %>%
                                      summarise(n_days = n(),
                                                dry_spell_rainfall = round(sum(total_prec), 1))

dry_spells_details <- dry_spells_list %>%
                        left_join(rainfall_during_dry_spells_stats, by = c('cell' = 'cell', 'dry_spell_confirmation' = 'dry_spell_confirmation', 'dry_spell_duration' = 'n_days'))

nrow(dry_spells_list) == nrow(dry_spells_details) # check that all records were kept

#write.csv(dry_spells_details, file = paste0(data_dir, "/processed/malawi/dry_spells/dry_spells_details_per_pixel.csv"), row.names = FALSE)

# identify dry spells during rainy seasons
dry_spells_details$cell <- as.character(dry_spells_details$cell)
dry_spells_during_rainy_season_list <- dry_spells_details %>%
                                          left_join(rainy_seasons[, c('cell', 'season_approx', 'onset_date', 'cessation_date')], by = c('cell', 'season_approx'), all.x = T, all.y = T) %>% # add rainy onset and cessation dates
                                          mutate(confirmation_date_during_rainy_season = ifelse(dry_spell_confirmation >= onset_date & dry_spell_confirmation <= cessation_date, 1, 0)) %>% # identifies dry spells that reached 14-d rolling sum during rainy season 
                                          filter(confirmation_date_during_rainy_season == 1) %>% # only keep dry spells that were confirmed during rainy season even if started before onset or ended after cessation
                                          dplyr::select(cell, season_approx, dry_spell_first_date, dry_spell_last_date, dry_spell_duration, dry_spell_rainfall)

# add adm names
dry_spells_during_rainy_season_list <- dry_spells_during_rainy_season_list %>% 
                                          left_join(cell_adms, by = c('cell'= 'cell')) %>%
                                          dplyr::select(cell, ADM2_EN, region, season_approx, dry_spell_first_date, dry_spell_last_date, dry_spell_duration, dry_spell_rainfall)


#write.csv(dry_spells_during_rainy_season_list, file = paste0(data_dir, "/processed/malawi/dry_spells/dry_spells_during_rainy_season_list_per_pixel.csv"), row.names = FALSE)

##############
# compute dry spell coverage per adm 
##############

# get cell counts per adm1 and adm2

adm2_cell_counts <- data_long %>%
                      dplyr::select(cell, ADM2_PCODE, ADM2_EN) %>%
                      group_by(ADM2_PCODE, ADM2_EN) %>%
                      summarise(n_cells = n_distinct(cell)) 
  
summary(adm2_cell_counts) # Note: 2 districts only include 1 cell

adm1_cell_counts <- data_long %>%
                      dplyr::select(cell) %>%
                      mutate(cell = as.character(cell)) %>%
                      left_join(cell_adms, by = c('cell'= 'cell')) %>%
                      group_by(region) %>%
                      summarise(n_cells = n_distinct(cell)) 

summary(adm1_cell_counts)

sum(adm1_cell_counts$n_cells) == sum(adm2_cell_counts$n_cells) # check for the same number of cells at adm2 and adm1

# roll up "dry spell cells" to adm1 and adm2 (all dry spells, regardless of whether they occured during rainy season)
all_dry_spells <- dry_spells_details %>%
                    left_join(cell_adms, by = c('cell'= 'cell')) %>%
                    dplyr::select(cell, ADM2_EN, region, season_approx, dry_spell_first_date, dry_spell_last_date, dry_spell_duration, dry_spell_rainfall)

all_dry_spells_adm2 <- all_dry_spells %>%
                        group_by(ADM2_EN) %>%
                        

############################

####### SCRATCHPAD BELOW

############################

# count number of pixels in total and below threshold per adm2
computePixelsPerADM2 <- function(layer, data_frame){
  
  # select 1 layer
  data_layer <- subset(s2001_cropped, layer)
  
  # extract values from raster cells and compute stat
  data_layer.stat <- raster::extract(data_layer, mwi_adm2, fun = function(x,...)length(x), df = T)
  
  # add daily stat to dataframe
  data_frame <- merge(data_frame, data_layer.stat, by = "ID", all.x = T)
  
  return(data_frame)
}

# loop through layers/days to compile number of values per adm2 across layers/days
value_count <- data.frame(ID = 1:nrow(mwi_adm2))

for (i in seq_along(1:5)) {
  
  value_count <- computePixelsPerADM2(i, value_count)
  
}


# count number of pixels below threshold per adm2
computePixelsBelowThreshold <- function(layer, data_frame, threshold){
  
  # select 1 layer
  data_layer <- subset(s2001_cropped, layer)
  
  # extract values from raster cells and compute stat
  data_layer.stat <- raster::extract(data_layer, mwi_adm2, fun = function(x,...)sum(x <= threshold), df = T)
  
  # add daily stat to dataframe
  data_frame <- merge(data_frame, data_layer.stat, by = "ID", all.x = T)
  
  return(data_frame)
}


# loop through layers/days to compile number of values below threshold across layers/days
values_below_th <- data.frame(ID = 1:nrow(mwi_adm2))

for (i in seq_along(1:5)) {
  
  values_below_th <- computePixelsBelowThreshold(i, values_below_th, threshold = 2)
  
}


#####
## explore rainy season patterns
#####

# onset, cessation, duration of rainy seasons by region
prop.table(table(rainy_seasons_detail$region, rainy_seasons_detail$onset_month), 1)
prop.table(table(rainy_seasons_detail$region, rainy_seasons_detail$cessation_month), 1)

ggplot(rainy_seasons_detail, aes(rainy_season_duration, fill = region)) +
  geom_histogram(binwidth = 10, position = "dodge") + 
  ylab("Number of cells*years") +
  xlab("Duration in Days") +
  ggtitle("Rainy Season Duration By Region")

# onset, cessation, duration of rainy seasons per district
prop.table(table(rainy_seasons_detail$ADM2_EN, rainy_seasons_detail$onset_month), 1)
prop.table(table(rainy_seasons_detail$ADM2_EN, rainy_seasons_detail$cessation_month), 1)

rainy_seasons_summary_per_region <- rainy_seasons_detail %>%
  mutate(nov1 = as.Date(paste0(season_approx, '-11-01'), format = "%Y-%m-%d"), # 1 nov before the onset of the season
         onset_days_since_nov1 = as.numeric(difftime(onset_date, nov1, units = "days")), # count of days since 1 nov
         cessation_days_since_nov1 = as.numeric(difftime(cessation_date, nov1, units = "days")), # count of days since 1 nov
         rainy_season_at_least_125d = ifelse(rainy_season_duration >= 125, 1, 0)) %>% # 125 days is length of maize growing season
  group_by(ADM2_EN) %>%
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

rainy_seasons_summary_per_region



#####
## explore rainy-season dry spells patterns
#####

# label region of the district
rainy_season_dry_spells_summary_per_region <- rainy_season_dry_spells_summary_per_region %>% mutate(region = substr(pcode, 3, 3)) %>% mutate(region = ifelse(region == 3, "Southern", ifelse(region == 2, "Central", "Northern")))

# how frequently have rainy-season dry spells occurred over the years and across regions/districts?
summary(rainy_season_dry_spells_summary_per_region$nbr_dry_spells)
prop.table(table(rainy_season_dry_spells_summary_per_region$nbr_dry_spells))

# how many and which districts/regions have not had a rainy-season dry spell?
rainy_season_dry_spells_summary_per_region %>% filter(nbr_dry_spells == 0) %>% summarise(n = n_distinct(ADM2_EN)) # districts
rainy_season_dry_spells_summary_per_region %>% filter(nbr_dry_spells == 0) %>% dplyr::select(ADM2_EN) %>% unique()

rainy_season_dry_spells_summary_per_region %>% filter(nbr_dry_spells == 0) %>% summarise(n = n_distinct(region)) # regions
rainy_season_dry_spells_summary_per_region %>% filter(nbr_dry_spells == 0) %>% dplyr::select(region) %>% unique()

# how many and which districts/regions have had a rainy-season dry spells?
rainy_season_dry_spells_summary_per_region %>% filter(nbr_dry_spells > 0) %>% summarise(n = n_distinct(ADM2_EN)) # districts
rainy_season_dry_spells_summary_per_region %>% filter(nbr_dry_spells > 0) %>% dplyr::select(ADM2_EN) %>% unique()

rainy_season_dry_spells_summary_per_region %>% filter(nbr_dry_spells > 0) %>% summarise(n = n_distinct(region)) # regions
rainy_season_dry_spells_summary_per_region %>% filter(nbr_dry_spells > 0) %>% dplyr::select(region) %>% unique()

# when did the dry spells start in each district?
prop.table(table(lubridate::month(dry_spells_during_rainy_season_list$dry_spell_first_date)))
prop.table(table(dry_spells_during_rainy_season_list$ADM2_EN, lubridate::month(dry_spells_during_rainy_season_list$dry_spell_first_date)), 1)

# when did the dry spells end in each district?
prop.table(table(lubridate::month(dry_spells_during_rainy_season_list$dry_spell_last_date)))
prop.table(table(dry_spells_during_rainy_season_list$ADM2_EN, lubridate::month(dry_spells_during_rainy_season_list$dry_spell_last_date)), 1)

####
# Viz
####
map_original_def <- mwi_adm2 %>% 
  left_join(rainy_season_dry_spells_summary_per_region, by = c('ADM2_PCODE' = 'pcode', 'ADM2_EN' = 'ADM2_EN')) %>%
  ggplot() +
  geom_sf(aes(fill = nbr_dry_spells)) +
  scale_fill_continuous(type = "viridis", "Number of dry spells",
                        breaks=c(0, 4, 8, 12),
                        limits=c(0,12)) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  ggtitle('Dry spells during 2000-2020', subtitle = "14-day period with cumulative rainfall <= 2mm") +
  labs(caption="Mean values per district, CHIRPS")

map_consec2mm_def <- mwi_adm2 %>% 
  left_join(daily_max_dry_spells_summary_per_region_2mm, by = c('ADM2_PCODE' = 'pcode', 'ADM2_EN' = 'ADM2_EN')) %>%
  ggplot() +
  geom_sf(aes(fill = nbr_dry_spells)) +
  scale_fill_continuous(type = "viridis", "Number of dry spells",
                        breaks=c(0, 4, 8, 12),
                        limits=c(0,12)) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  ggtitle('Dry spells during 2000-2020', subtitle = "14 consecutive days with <= 2mm rainfall daily") +
  labs(caption="Mean values per district, CHIRPS")

map_consec4mm_def <- mwi_adm2 %>% 
  left_join(daily_max_dry_spells_summary_per_region, by = c('ADM2_PCODE' = 'pcode', 'ADM2_EN' = 'ADM2_EN')) %>%
  ggplot() +
  geom_sf(aes(fill = nbr_dry_spells)) +
  scale_fill_continuous(type = "viridis", "Number of dry spells", 
                        breaks=c(0, 4, 8, 12),
                        limits=c(0,12)) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  ggtitle('Dry spells during 2000-2020', subtitle = "14 consecutive days with <= 4mm rainfall daily") +
  labs(caption="Mean values per district, CHIRPS")

#gridExtra::grid.arrange(map_original_def, map_consec2mm_def, map_consec4mm_def, nrow = 1) # for viewing in console only
grob <- gridExtra::arrangeGrob(map_original_def, map_consec2mm_def, map_consec4mm_def, nrow = 1) # creates object that can be saved programmatically
#ggsave(file=paste0(data_dir, "/processed/malawi/dry_spells/dry_spell_plots/definition_comparison.png"), grob)

# save list of dry spells
#write.csv(dry_spells_during_rainy_season_list, file = paste0(data_dir, "/processed/malawi/dry_spells/dry_spells_during_rainy_season_list_2000_2020_per_pixel.csv"), row.names = FALSE)

#full_list_dry_spells <- dry_spells_details %>%
#                         left_join(rainy_seasons[, c('cell', 'season_approx', 'onset_date', 'cessation_date')], by = c('cell', 'season_approx'), all.x = T, all.y = T) %>% # add rainy onset and cessation dates
#                        mutate(confirmation_date_during_rainy_season = ifelse(dry_spell_confirmation >= onset_date & dry_spell_confirmation <= cessation_date, 1, 0)) %>% # identifies dry spells that reached 14-d rolling sum during rainy season 
#                       dplyr::select(cell, season_approx, dry_spell_first_date, dry_spell_last_date, dry_spell_duration, dry_spell_rainfall)
#write.csv(full_list_dry_spells, file = paste0(data_dir, "/processed/malawi/dry_spells/full_list_dry_spells.csv"), row.names = FALSE)

# summary stats per region 
rainy_season_dry_spells_summary_per_region <- dry_spells_during_rainy_season_list %>% 
  group_by(pcode, ADM2_EN) %>%
  summarise(nbr_dry_spells = n(),
            mean_ds_duration = round(mean(dry_spell_duration),1),
            min_ds_duration = min(dry_spell_duration),
            max_ds_duration = max(dry_spell_duration)
  ) %>%
  ungroup() %>%
  as.data.frame()

rainy_season_dry_spells_summary_per_region <- merge(rainy_season_dry_spells_summary_per_region, mwi_adm2_ids, by.x = c('pcode', 'ADM2_EN'), by.y = c('ADM2_PCODE', 'ADM2_EN'), all.y = T) # ensure every region is in dataset
rainy_season_dry_spells_summary_per_region$nbr_dry_spells <- ifelse(is.na(rainy_season_dry_spells_summary_per_region$nbr_dry_spells), 0, rainy_season_dry_spells_summary_per_region$nbr_dry_spells) # replace NAs with 0 under nbr of dry spells

rainy_season_dry_spells_summary_per_region
  