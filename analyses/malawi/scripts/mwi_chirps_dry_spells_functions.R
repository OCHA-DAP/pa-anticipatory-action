
computeLayerStat <- function(layer, stat, data_stat_values){
  
  # select 1 layer
  data_layer <- subset(data, layer)
  
  # extract values from raster cells and compute stat
  data_layer.stat <- raster::extract(data_layer, mwi_adm2, fun = stat, df = T)
  
  # add daily stat to dataframe
  data_stat_values <- merge(data_stat_values, data_layer.stat, by = "ID", all.x = T)
  
  return(data_stat_values)
}

convertToLongFormat <- function(data.wideformat){
  
  # add pcodes to identify each polygon
  data.wideformat$pcode <- mwi_adm2$ADM2_PCODE
  
  # convert wide to long to get dates as rows
  data.longformat <- gather(data.wideformat, date, total_prec, 2:(nbr_layers+1))
  
  # assign "zero" values to NA in total_prec
  data.longformat$total_prec[is.na(data.longformat$total_prec)] <- 0
  
  # reformat 'date' to a date format
  data.longformat$date <- as.Date(data.longformat$date, format = "X%Y.%m.%d")
  
  return(data.longformat)
}

## compute rolling sum
computeRollingSum <- function(dataframe_long, window){
  
    # convert to wide
    rolling_sum <-  dataframe_long %>%
          arrange(pcode, date) %>%
          group_by(pcode) %>%
          mutate(rollsum = zoo::rollsum(total_prec, k = window, fill = NA, align = 'right')
    ) 
  return(rolling_sum)
}

# compute backwards rolling sum
computeBackRollingSum <- function(dataframe_long, window){
  
  # convert to wide
  rolling_sum <-  dataframe_long %>%
    arrange(pcode, date) %>%
    group_by(pcode) %>%
    mutate(rollsum = zoo::rollsum(total_prec, k = window, fill = NA, align = 'left')
    ) 
  return(rolling_sum)
}


## compute onset date for every rainy season per adm2
findRainyOnset <- function() {
  
  # identify 10-day periods with at least 40mm cumulative yearound
  data_max_values_long$min_cum_40mm_bin <- ifelse(data_max_values_long$rollsum_10d >= 40, 1, 0) # is this day in a 40+mm period?
  
  # identify 10-day dry spells (10 consecutive days with less than 2mm of total rain) yearound
  data_max_values_long$less_than_cum_2mm_bin <- ifelse(data_max_values_long$rollsum_10d < 2, 1, 0) # is this day in a dry period (<2mm)?
  
  # verify no 10-day dry spells following in 30 days of first day with >=40mm cum sum yearound
  data_max_values_long <- data_max_values_long %>% 
                            group_by(pcode) %>%
                            mutate(nbr_dry_spell_days_win_30d = zoo::rollsum(less_than_cum_2mm_bin, k = 30, fill = NA, align = 'left'),
                                   followed_by_ds_win_30d_bin = ifelse(nbr_dry_spell_days_win_30d > 0, 1, 0))  
                          
  # select earliest date per season_approx after 1 Nov that meets both criteria
  rainy_onsets <- data_max_values_long %>%
                    filter(season_approx != 'outside rainy season') %>% # exclude Aug-Sept
                    mutate(meets_onset_criteria = ifelse(min_cum_40mm_bin == 1 & followed_by_ds_win_30d_bin == 0, 1, 0)) %>% # min 40mm in 15 days and not followed by 10d dry spells within 30 days
                    filter(meets_onset_criteria == 1 & month %in% c(11, 12, 1, 2)) %>% # period post 1 Nov. Allows Nov-Feb for onsets)
                    group_by(pcode, season_approx) %>%
                    slice(which.min(date)) %>%
                    ungroup() %>%
                    dplyr::select(ID, pcode, season_approx, date) %>%
                    rename(onset_date = date) 
                    
  rainy_onsets$onset_date[rainy_onsets$season_approx == '1999'] <- NA # set values for 1999 season as NA since no data Oct-Dec 1999 available
  
  return(rainy_onsets)
  
          }

## compute cessation date for every rainy season per adm2
findRainyCessation <- function() {
  
   # identify 15-day periods of up to 25mm cum
    data_max_values_long$max_cum_25mm_bin <- ifelse(data_max_values_long$rollsum_15d <= 25, 1, 0) # is this day in a 25mm or less 15d period?
  
   # select earliest date per season_approx after 15 March that meets criterion
    
    # select earliest date per season_approx after 15 March that meets criterion
    rainy_cessation <- data_max_values_long %>%
                          filter((month >= 4 & month < 8) | (month == 3 & day >= 15)) %>% # in Mar on or after the 15th, or between April and Aug exclusively
                          filter(max_cum_25mm_bin == 1) %>% # meet criterion for cessation (= 25mm or less cumulative rainfall over 15 days)
                          group_by(pcode, season_approx) %>%
                          slice(which.min(date)) %>%
                          ungroup() %>%
                          dplyr::select(ID, pcode, season_approx, date) %>%
                          rename(cessation_date = date) 
    
    return(rainy_cessation)

}

## user-defined run-length encoding function in base R
runlengthEncoding <- function(x) {
  x <- rle(x)$lengths
  rep(seq_along(x), times=x)
}











