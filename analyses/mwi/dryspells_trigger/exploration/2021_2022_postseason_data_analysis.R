# analyse precipitation data for postseason analysis of 2021-2022 season
# this script sources 2021_2022_postseason_data_pull.R
# this script is sourced by 2021_2022_postseason_overview.Rmd

#####
## setup
#####

# load data and useful R objects
source('2021_2022_postseason_data_pull.R')

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

## TO DO handle rainfall = NAs ****

## 
# Rainy season stats
##

# identify streaks of rainy days per cell (streak minimum: 1 day)
stats <- lf %>%
        group_by(cell) %>%
        arrange(dates) %>%
        mutate(lagged = dplyr::lag(rainy_bin),
               streak_start = case_when(row_number() == 1 & !is.na(rainfall) ~ TRUE  # If cell's first date and there is rainfall data, then it's first day of a streak. If not, check if this day's value is the same as previous row's.
                                      ,row_number() == 1 & is.na(rainfall) ~ NA
                                      ,TRUE ~ (rainy_bin != lagged)),
               streak_id = ifelse(!is.na(streak_start), cumsum(streak_start), NA),
               streak_type = ifelse(rainy_bin == 0, "dry", "wet")) %>%
        group_by(cell, streak_id) %>%
        mutate(day_into_streak = ifelse(!is.na(streak_id), row_number(), NA)) %>%
        ungroup() %>%
        arrange(cell, dates)

## identify season onset
# First day of a period after 1 Nov with at least 40mm of rain over 10 days 
# AND no 10 consecutive days with less than 2mm of total rain in the following 30 days (DCCMS 2008).
#stats$roll_sum_prev_10d <- ifelse(!is.na(stats$rainfall), zoo::rollsum(stats$rainfall, k = 10, fill = NA, align = 'right'), NA) # sum of precipitation in previous 10 days
stats$roll_sum_foll_10d <- ifelse(!is.na(stats$rainfall), zoo::rollsum(stats$rainfall, k = 10, fill = NA, align = 'left'), NA) # sum of precipitation in following 10 days
stats$foll_by_10d_ds_bin <- ifelse(stats$roll_sum_foll_10d < 2, 1, 0)
  
onsets <- stats %>%
             group_by(cell) %>%
             arrange(dates) %>%
             mutate(followed_by_ds = ifelse(!is.na(rainfall), zoo::rollsum(foll_by_10d_ds_bin, k = 20, align = 'left') >= 1, NA)) %>% # boolean: if at least 1 day is followed by 10-day cum sum less than 2. 20th day last chance to be followed by 10 days within 30-day period
             filter(dates >= '2021-11-01' & roll_sum_foll_10d >= 40 & followed_by_ds == FALSE) %>% # after 1 Nov, at least 40mm of rain over 10 days since 1 Nov and not followed by 10 days with 2mm or less of rain within 30 days
             slice(which.min(dates)) %>% # retrieve earliest date that meets criterion per cell 
             ungroup() %>%
             select(cell, date_chr, dates, roll_sum_foll_10d, followed_by_ds) %>%
             rename(onset_date_chr = date_chr, onset_date = dates) %>%
             mutate(onset_days_since_1nov = as.numeric(difftime(onset_date, as.Date("2021-11-01"), unit = "days"))) %>% # Day 0 = 1 Nov
             as.data.frame()

#####
# create raster with results
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





