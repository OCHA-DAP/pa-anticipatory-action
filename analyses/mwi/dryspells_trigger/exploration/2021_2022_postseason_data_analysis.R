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
labels <- as.data.frame(seq(from = as.Date('2021-10-01', format = '%Y-%m-%d'), 
                   to = as.Date('2022-03-20', format = '%Y-%m-%d'), ## FIX ME
                   # to = as.Date('2022-04-01', format = '%Y-%m-%d'), 
                   by = "days"))

dates_df <- as.data.frame(dates) 
date_labels <- cbind(dates_df, labels)

# pivot data to long format
lf <- data %>% 
        select(-ID, -x, -y) %>%
        pivot_longer(
          cols = starts_with("est_prcp_T="),
          names_to = "date_chr",
          values_to = "rainfall"
          #values_drop_na = TRUE
        ) %>%
        left_join(date_labels, by = c('date_chr' = 'dates')) 

# identify week and month
lf$month <- lubridate::month(lf$labels)
lf$week <- lubridate::week(lf$labels)

lf <- lf %>%
      select(cell, date_chr, labels, month, week, rainfall)  

lf <- lf %>% arrange(cell, labels) 

###
# analysis
###

## TO DO handle rainfall = NAs ****

# determine if day was rainy. Binary value. Rainy = received 3 or more mm of rain. Definition from February 2022 bulletin from DCCMS
lf$rainy_bin <- ifelse(lf$rainfall >= 3, 1, 0)

# identify streaks of rainy days per cell (streak minimum: 1 day)
debug <- lf %>%
        group_by(cell) %>%
        mutate(lagged = dplyr::lag(rainy_bin),
               streak_start = ifelse(row_number() == 1, TRUE, (rainy_bin != lagged)),  # If first date of cell, then it's first day of a streak. If not, check if this day's value is the same as previous row's.
               streak_id = cumsum(streak_start)) %>%
        group_by(cell, streak_id) %>%
        mutate(day_into_streak = ifelse(!is.na(streak_id), row_number(), NA)) %>%
        ungroup() %>%
        mutate(streak_type = ifelse(rainy_bin == 0, "dry", "wet"))


















