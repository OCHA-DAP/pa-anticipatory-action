# analyse precipitation data for postseason analysis of 2021-2022 season
# this script sources 2021_2022_postseason_data_pull.R
# this script is sourced by 2021_2022_postseason_overview.Rmd

#####
## data pull
#####

# load data and useful R objects
source('dryspells_trigger/exploration/2021_2022_postseason_data_pull.R')

#####
## data formatting
#####

# user-friendly date labels
dates <- data.frame(dates = seq(from = as.Date('2021-10-01', format = '%Y-%m-%d'), 
                   to = as.Date('2022-04-30', format = '%Y-%m-%d'), 
                   by = "days"))

date_labels <- cbind(dates_chr, dates)

# pivot data to long format & formatting
lf <- data %>% 
        select(-ID, -x, -y) %>%
        pivot_longer(
          cols = starts_with("est_prcp_T="),
          names_to = "date_chr",
          values_to = "rainfall"
          #values_drop_na = TRUE
        ) %>%
        left_join(date_labels, by = c('date_chr' = 'dates_chr')) %>%
        mutate(month = lubridate::month(dates), # identify calendar month
               week = lubridate::week(dates), # identify calendar week
               rainy_bin = ifelse(rainfall >= 3, 1, 0)) %>% # binary: was >= 3mm of rain received? Rainy day definition from February 2022 bulletin from DCCMS
      select(cell, date_chr, dates, month, week, rainfall, rainy_bin) %>%
      arrange(cell, dates) 

#####
# analysis
#####

## 
# Rainy season stats
##

# compute rolling sums
stats <- lf %>%
  group_by(cell) %>%
  arrange(cell, dates) %>%
  dplyr::mutate(roll_sum_next_10d = ifelse(!is.na(rainfall), zoo::rollsum(rainfall, k = 10, fill = NA, align = 'left'), NA), # sum of precipitation in 10-day period starting today
         start_10d_ds_bin = ifelse(roll_sum_next_10d < 2, 1, 0), # is this the first day of a dry 10-day period?
         roll_sum_foll_15d = ifelse(!is.na(rainfall), zoo::rollsum(lead(rainfall, 1), k = 15, fill = NA, align = 'left'), NA), # sum of precipitation in 15-day period starting tomorrow (lead=1)
         foll_by_15d_ds_bin = ifelse(roll_sum_foll_15d <= 25, 1, 0), # is tomorrow the first day of a 15-d dry period?
         roll_sum_next_14d = ifelse(!is.na(rainfall), zoo::rollsum(rainfall, k = 14, fill = NA, align = 'left'), NA), # sum of precipitation in 14-day period starting today 
         )

# Onsets:
# First day of a period after 1 Nov with at least 40mm of rain over 10 days 
# AND no 10 consecutive days with less than 2mm of total rain in the following 30 days (DCCMS 2008).
onsets <- stats %>%
             group_by(cell) %>%
             arrange(cell, dates) %>%
             mutate(followed_by_ds = ifelse(!is.na(rainfall), zoo::rollsum(lead(start_10d_ds_bin, 1), k = 20, align = 'left') >= 1, NA)) %>% # boolean: is there at least one 10-day period with cum sum less than 2 in the next 20 days starting tomorrow (lead = 1)? k=20th day is the last chance to start a 10d-period within 30-day period
             filter(dates >= '2021-11-01' & roll_sum_next_10d >= 40 & followed_by_ds == FALSE) %>%
             slice(which.min(dates)) %>% # retrieve earliest date that meets criterion per cell 
             ungroup() %>%
             select(cell, date_chr, dates, roll_sum_next_10d, followed_by_ds) %>%
             rename(onset_date_chr = date_chr, onset_date = dates) %>%
             mutate(onset_days_since_1nov = as.numeric(difftime(onset_date, as.Date("2021-11-01"), unit = "days"))) %>% # Day 0 = 1 Nov
             as.data.frame()

# Cessations:
# Last day before a 15-day period after 15 March with 25mm or less of rain
cessations <- stats %>%
            group_by(cell) %>%
            arrange(cell, dates) %>%
            filter(dates >= '2022-03-15' & foll_by_15d_ds_bin == TRUE) %>%  # on or after 15 March
            slice(which.min(dates)) %>% # retrieve earliest date that meets criterion per cell 
            ungroup() %>%
            select(cell, date_chr, dates, roll_sum_foll_15d, foll_by_15d_ds_bin) %>%
            rename(cessation_date_chr = date_chr, cessation_date = dates) %>%
            mutate(cessation_days_since_1nov = as.numeric(difftime(cessation_date, as.Date("2021-11-01"), unit = "days"))) %>% # Day 0 = 1 Nov
            as.data.frame()

# Duration:
season_dates <- data.frame(cell_numbers) %>%
                    select(cell) %>%
                    left_join(onsets, by = 'cell') %>%
                    left_join(cessations, by = 'cell') %>%
                    select(cell, onset_date, onset_days_since_1nov, cessation_date, cessation_days_since_1nov) %>%
                    mutate(duration = as.numeric(cessation_date - onset_date))

#####
## in-season analysis
#####

#in_season <- lf %>%
in_season <- stats %>%
       left_join(season_dates[, c('cell', 'onset_date', 'cessation_date', 'duration')], by = 'cell') %>%
       mutate(in_seas_bin = case_when(is.na(onset_date) | is.na(cessation_date) ~ "unknown",
                                      dates >= onset_date & dates <= cessation_date ~ "1",
                                      TRUE ~ "0")) %>% # 1 and 0 are strings because of 'UNKNOWN'
       filter(in_seas_bin == '1')  # keep in-season data 

          
# identify in-season streaks of rainy days per cell (streak minimum: 1 day)
streaks <- in_season %>%
            group_by(cell) %>%
            arrange(cell, dates) %>%
            mutate(lagged = dplyr::lag(rainy_bin),
                   streak_start = case_when(row_number() == 1 & !is.na(rainfall) ~ TRUE,  # If cell's first date and there is rainfall data, then it's first day of a streak. If not, check if this day's value is the same as previous row's.
                                            row_number() == 1 & is.na(rainfall) ~ NA,
                                            TRUE ~ (rainy_bin != lagged)),
                   streak_id = ifelse(!is.na(streak_start), cumsum(streak_start), NA),
                   streak_type = ifelse(rainy_bin == 0, "dry", "wet")) %>%
            group_by(cell, streak_id) %>%
            mutate(day_into_streak = ifelse(!is.na(streak_id), row_number(), NA)) %>%
            ungroup() %>%
            arrange(cell, dates)

# identify longest dry streak per cell
max_dry_streak <- streaks %>%
              filter(streak_type == 'dry') %>%
              group_by(cell) %>%
              slice(which.max(day_into_streak)) %>%
              ungroup() %>%
              select(cell, day_into_streak) %>%
              rename(max_streak = day_into_streak)

# count number of dry streaks per cell
dry_5dplus_streak_count <- streaks %>%
                  filter(streak_type == 'dry' & day_into_streak >= 5) %>%
                  group_by(cell) %>%
                  summarise(n_5dplus_streaks = n_distinct(streak_id)) %>%
                  ungroup() %>%
                  select(cell, n_5dplus_streaks)

dry_7dplus_streak_count <- streaks %>%
  filter(streak_type == 'dry' & day_into_streak >= 7) %>%
  group_by(cell) %>%
  summarise(n_7dplus_streaks = n_distinct(streak_id)) %>%
  ungroup() %>%
  select(cell, n_7dplus_streaks)

dry_10dplus_streak_count <- streaks %>%
  filter(streak_type == 'dry' & day_into_streak >= 10) %>%
  group_by(cell) %>%
  summarise(n_10dplus_streaks = n_distinct(streak_id)) %>%
  ungroup() %>%
  select(cell, n_10dplus_streaks)

dry_14dplus_streak_count <- streaks %>%
  filter(streak_type == 'dry' & day_into_streak >= 14) %>%
  group_by(cell) %>%
  summarise(n_14dplus_streaks = n_distinct(streak_id)) %>%
  ungroup() %>%
  select(cell, n_14dplus_streaks)

dry_streak_counts <- dry_5dplus_streak_count %>%
                    full_join(dry_7dplus_streak_count, by = 'cell', ) %>%
                    full_join(dry_10dplus_streak_count, by = 'cell') %>%
                    full_join(dry_14dplus_streak_count, by = 'cell')
  
# identify in-season rainy days
rainy_day_counts <- in_season %>%
               group_by(cell) %>%
               summarise(rainy_days_n = sum(rainy_bin))

rainy_ratio <- in_season %>%
               select(cell, duration) %>%
               distinct() %>%
               left_join(rainy_day_counts, by = 'cell') %>%
               mutate(rainy_days_perc = 100 * round(rainy_days_n / duration, 2))

# compute in-season rainy totals
rain_seas_totals <- in_season %>%
  group_by(cell) %>%
  summarise(rainfall_seas_total = sum(rainfall)) %>%
  ungroup()

# compute number of dry days per month, regardless of whether season has begun/ended
dry_days_per_month <- stats %>%
                      group_by(cell, month) %>%
                      summarise(dry_days_n = sum(ifelse(rainy_bin == 0, 1, 0), na.rm = T)) %>%
                      ungroup() %>%
                      mutate(month = recode(month, `1` = 'jan', `2` = 'feb', `3` =  'mar', `4` = 'apr', `10` = 'oct', `11` = 'nov', `12` = 'dec')) %>%
                      spread(month, dry_days_n)

#####
# create raster files with results
#####

# create blank raster and cell number list
raster_template <- terra::subset(data_r, 1) # keep a single layer and create a template raster
template_cells <- data.frame(cell = 1:ncell(raster_template))

names(raster_template) <- "discardable" # rename existing layer
varnames(raster_template) <- "discardable" # rename existing variable

# create time-static raster
static_r <- raster_template

# create onset raster layer
season_all_cells <- left_join(template_cells, season_dates, by = 'cell')

onset_r <- raster_template
onset_r <- setValues(onset_r, season_all_cells$onset_days_since_1nov)
varnames(onset_r) <- "onset"
names(onset_r) <- "onset"

# create cessation raster layer
cessation_r <- raster_template
cessation_r <- setValues(cessation_r, season_all_cells$cessation_days_since_1nov)
varnames(cessation_r) <- "cessation"
names(cessation_r) <- "cessation"

# create duration raster layer
duration_r <- raster_template
duration_r <- setValues(duration_r, season_all_cells$duration)
varnames(duration_r) <- "duration"
names(duration_r) <- "duration"

# create rainy_ratio raster layer
rainy_ratio_all_cells <- left_join(template_cells, rainy_ratio, by = 'cell')
rainy_ratio_r <- raster_template
rainy_ratio_r <- setValues(rainy_ratio_r, rainy_ratio_all_cells$rainy_days_perc)
varnames(rainy_ratio_r) <- "rainy_ratio"
names(rainy_ratio_r) <- "rainy_ratio"

# create rain season total raster layer
rain_seas_tot_all_cells <- left_join(template_cells, rain_seas_totals, by = 'cell')
rain_seas_tot_r <- raster_template
rain_seas_tot_r <- setValues(rain_seas_tot_r, rain_seas_tot_all_cells$rainfall_seas_total)
varnames(rain_seas_tot_r) <- "rain_seas_tot"
names(rain_seas_tot_r) <- "rain_seas_tot"

# create max dry streak raster layer
max_dry_streak_all_cells <- left_join(template_cells, max_dry_streak, by = 'cell')
max_streak_r <- raster_template
max_streak_r <- setValues(max_streak_r, max_dry_streak_all_cells$max_streak)
varnames(max_streak_r) <- "max_streak"
names(max_streak_r) <- "max_streak"

# create dry streak count raster layers
dry_streak_counts_all_cells <- left_join(template_cells, dry_streak_counts, by = 'cell')
n_5dplus_streaks_r <- raster_template
n_5dplus_streaks_r <- setValues(n_5dplus_streaks_r, dry_streak_counts_all_cells$n_5dplus_streaks)
varnames(n_5dplus_streaks_r) <- "n_5dplus_streaks"
names(n_5dplus_streaks_r) <- "n_5dplus_streaks"

n_7dplus_streaks_r <- raster_template
n_7dplus_streaks_r <- setValues(n_7dplus_streaks_r, dry_streak_counts_all_cells$n_7dplus_streaks)
varnames(n_7dplus_streaks_r) <- "n_7dplus_streaks"
names(n_7dplus_streaks_r) <- "n_7dplus_streaks"

n_10dplus_streaks_r <- raster_template
n_10dplus_streaks_r <- setValues(n_10dplus_streaks_r, dry_streak_counts_all_cells$n_10dplus_streaks)
varnames(n_10dplus_streaks_r) <- "n_10dplus_streaks"
names(n_10dplus_streaks_r) <- "n_10dplus_streaks"

n_14dplus_streaks_r <- raster_template
n_14dplus_streaks_r <- setValues(n_14dplus_streaks_r, dry_streak_counts_all_cells$n_14dplus_streaks)
varnames(n_14dplus_streaks_r) <- "n_14dplus_streaks"
names(n_14dplus_streaks_r) <- "n_14dplus_streaks"

dry_days_per_month_r <- c(rep(raster_template, 6))
varnames(dry_days_per_month_r) <- 'dry_days_n'
names(dry_days_per_month_r) <- c('nov', 'dec', 'jan', 'feb', 'mar', 'apr')
dry_days_per_month_all_cells <- left_join(template_cells, dry_days_per_month, by = 'cell')
dry_days_per_month_r[[1]][dry_days_per_month_all_cells$cell] <- dry_days_per_month_all_cells$nov
dry_days_per_month_r[[2]][dry_days_per_month_all_cells$cell] <- dry_days_per_month_all_cells$dec
dry_days_per_month_r[[3]][dry_days_per_month_all_cells$cell] <- dry_days_per_month_all_cells$jan
dry_days_per_month_r[[4]][dry_days_per_month_all_cells$cell] <- dry_days_per_month_all_cells$feb
dry_days_per_month_r[[5]][dry_days_per_month_all_cells$cell] <- dry_days_per_month_all_cells$mar
dry_days_per_month_r[[6]][dry_days_per_month_all_cells$cell] <- dry_days_per_month_all_cells$apr

### create rasters

# create static raster
static_r <- c(static_r, 
              onset_r, 
              cessation_r, 
              duration_r, 
              rainy_ratio_r,
              rain_seas_tot_r,
              max_streak_r,
              n_5dplus_streaks_r,
              n_7dplus_streaks_r,
              n_10dplus_streaks_r,
              n_14dplus_streaks_r)

# save datasets
saveRDS(object = in_season, file = paste0(dry_spell_processed_path, "2021_2022_postseason/" , "in_season.RDS"))
saveRDS(object = season_dates, file = paste0(dry_spell_processed_path, "2021_2022_postseason/" , "season_dates.RDS"))
saveRDS(object = streaks, file = paste0(dry_spell_processed_path, "2021_2022_postseason/" , "streaks.RDS"))

# save static rasters
writeRaster(raster_template, filename = paste0(dry_spell_processed_path, "2021_2022_postseason/raster_template.tif"), overwrite=T)
writeRaster(static_r, filename = paste0(dry_spell_processed_path, "2021_2022_postseason/static_r.tif"), overwrite=T)

# save time series raster
writeRaster(dry_days_per_month_r , filename = paste0(dry_spell_processed_path, "2021_2022_postseason/dry_days_per_month_r.tif"), overwrite=T)

