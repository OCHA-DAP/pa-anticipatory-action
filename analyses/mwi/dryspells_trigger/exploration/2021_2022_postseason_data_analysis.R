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
