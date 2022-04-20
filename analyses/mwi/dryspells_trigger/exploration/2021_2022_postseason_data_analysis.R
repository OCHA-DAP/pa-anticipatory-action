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
                   to = as.Date('2022-03-20', format = '%Y-%m-%d'), ## FIX ME
                   # to = as.Date('2022-04-01', format = '%Y-%m-%d'), 
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
         foll_by_15d_ds_bin = ifelse(roll_sum_foll_15d <= 25, 1, 0) # is tomorrow the first day of a 15-d dry period?
  )

# Onsets:
# First day of a period after 1 Nov with at least 40mm of rain over 10 days 
# AND no 10 consecutive days with less than 2mm of total rain in the following 30 days (DCCMS 2008).
onsets <- stats %>%
             group_by(cell) %>%
             arrange(dates) %>%
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
            arrange(dates) %>%
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

dat <- lf %>%
       left_join(season_dates[, c('cell', 'onset_date', 'cessation_date')], by = 'cell') %>%
       mutate(in_seas_bin = case_when(is.na(onset_date) | is.na(cessation_date) ~ "unknown",
                                      dates >= onset_date & dates <= cessation_date ~ "1",
                                      TRUE ~ "0")) %>% # 1 and 0 are strings because of 'UNKNOWN'
       filter(in_seas_bin == '1' | in_seas_bin == 'unknown') %>% # keep in-season data & keep cells without onset/cessation dates
       filter(dates >= "2021-11-01" & dates <= "2022-03-15") # arbitrary cutoffs for cells without onset/cessation dates
          

# identify in-season streaks of rainy days per cell (streak minimum: 1 day)
streaks <- dat %>%
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

#####
# create raster files with results
#####

# create blank raster and cell number list
raster_template <- terra::subset(data_r, 1) # keep a single layer and create a template raster
template_cells <- data.frame(cell = 1:ncell(raster_template))

names(raster_template) <- "discardable" # rename existing layer
varnames(raster_template) <- "discardable" # rename existing variable

# create time-static raster
static <- raster_template

# create onset raster layer
onsets_all_cells <- left_join(template_cells, onsets, by = 'cell')
onset_r <- raster_template
onset_r <- setValues(onset_r, onsets_all_cells$onset_days_since_1nov)
varnames(onset_r) <- "onset"
names(onset_r) <- "onset"

static <- c(static, onset_r)

## save results ##
saveRDS(object = season_dates, file = paste0(dry_spell_processed_path, "2021_2022_postseason/" , "season_dates.RDS"))
saveRDS(object = streaks, file = paste0(dry_spell_processed_path, "2021_2022_postseason/" , "streaks.RDS"))
writeRaster(raster_template, filename = paste0(dry_spell_processed_path, "2021_2022_postseason/raster_template.tif"))
writeRaster(static, filename = paste0(dry_spell_processed_path, "2021_2022_postseason/static_r.tif"))

