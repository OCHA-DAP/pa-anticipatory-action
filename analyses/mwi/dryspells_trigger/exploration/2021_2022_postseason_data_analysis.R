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
dates = seq(from = as.Date('2021-10-01', format = '%Y-%m-%d'), 
                   to = as.Date('2022-03-20', format = '%Y-%m-%d'), ## FIX ME
                   # to = as.Date('2022-04-01', format = '%Y-%m-%d'), 
                   by = "days")
dates_df <- as.data.frame(dates) 

date_labels <- cbind(dates_chr, dates_df)

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
# Season stats
##
Rainy season onset: 
First day of a period after 1 Nov with at least 40mm of rain over 10 days 
AND no 10 consecutive days with less than 2mm of total rain in the following 30 days (DCCMS 2008).



# identify streaks of rainy days per cell (streak minimum: 1 day)
stats <- lf %>%
        group_by(cell) %>%
        arrange(labels) %>%
        mutate(lagged = dplyr::lag(rainy_bin),
               streak_start = ifelse(row_number() == 1, TRUE, (rainy_bin != lagged)),  # If first date of cell, then it's first day of a streak. If not, check if this day's value is the same as previous row's.
               streak_id = cumsum(streak_start)) %>%
        group_by(cell, streak_id) %>%
        mutate(day_into_streak = ifelse(!is.na(streak_id), row_number(), NA)) %>%
        ungroup() %>%
        mutate(streak_type = ifelse(rainy_bin == 0, "dry", "wet"))



## identify season onset

## compute rolling sum
computeRollingSumPerPixel <- function(dataframe_long, window){
  
  rolling_sum <-  dataframe_long %>%
    arrange(cell, date) %>%
    group_by(cell) %>%
    mutate(rollsum = zoo::rollsum(total_prec, k = window, fill = NA, align = 'right')
    ) 
  return(rolling_sum)
}
## redo streaks within season


#####
# create new raster
#####

raster_template <- terra::subset(data_r, 1) # keep a single layer and create a template raster
names(raster_template) <- "discardable" # rename layers
varnames(raster_template) <- "varname" # rename variables

addValuestoRaster <- function(raster, df, col) {
  values(raster)[df[["cell"]]] <- df[[col]]
}









