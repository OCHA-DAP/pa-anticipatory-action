
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

## user-defined run-length encoding function in base R
runlengthEncoding <- function(x) {
     x <- rle(x)$lengths
     rep(seq_along(x), times=x)
}

## compute onset date for every rainy season per adm2
findRainyOnset <- function() {
  
  # get periods with at least 40mm of rain over 10 days
  data_max_values_long <- data_max_values_long %>%
                        group_by(pcode) %>%
                        computeRollingSum(., window = 10) %>%
                        rename(rollsum_10d = rollsum)
  
  data_max_values_long$min_cum_40mm_bin <- ifelse(data_max_values_long$rollsum_10d >= 40, 1, 0) # is this day in a 40+mm period?
  
  # identify 10-day dry spells (10 consecutive days with less than 2mm of total rain)
  data_max_values_long$less_than_cum_2mm_bin <- ifelse(data_max_values_long$rollsum_10d < 2, 1, 0) # is this day in a dry period (<2mm)?
  
  # verify no 10-day dry spells following in 30 days of first day with >=40mm cum sum
  data_max_values_long <- data_max_values_long %>% 
                            group_by(pcode) %>%
                            mutate(nbr_dry_spell_days_win_30d = zoo::rollsum(less_than_cum_2mm_bin, k = 30, fill = NA, align = 'left'),
                                   followed_by_ds_win_30d_bin = ifelse(nbr_dry_spell_days_win_30d > 0, 1, 0))  
                          
  # select earliest date per season_approx after 1 Nov that meets both criteria
  rainy_onsets <- data_max_values_long %>%
                    filter(season_approx != 'outside rainy season') %>% 
                    mutate(meets_onset_criteria = ifelse(min_cum_40mm_bin == 1 & followed_by_ds_win_30d_bin == 0, 1, 0)) %>% 
                    group_by(pcode, season_approx) %>%
                    filter(meets_onset_criteria == 1 & month != 10) %>% # period post 1 Nov. Excludes Oct but includes Jan for each season_approx)
                    slice(which.min(date)) %>%
                    ungroup() %>%
                    dplyr::select(ID, pcode, season_approx, date) %>%
                    rename(onset_date = date) %>% 
                    filter(season_approx != '2009') # excluding rainy season 2009 for incomplete data
  
  return(rainy_onsets)
  
          }

## compute cessation date for every rainy season per adm2
findRainyCessation <- function() {
  
  # get periods with at least 40mm of rain over 10 days
  data_max_values_long <- data_max_values_long %>%
    group_by(pcode) %>%
    computeRollingSum(., window = 15) %>%
    rename(rollsum_15d = rollsum)
  
  data_max_values_long$max_cum_25mm_bin <- ifelse(data_max_values_long$rollsum_15d <= 25, 1, 0) # is this day in a 25mm or less period?
  
  # select earliest date per season_approx after 15 March that meets criterion
  rainy_cessation <- data_max_values_long %>%
                        filter(season_approx != 'outside rainy season' & month >= 3 & day >= 15) %>% # rainy season but period post 15 March
                        group_by(pcode, season_approx) %>%
                        slice(which.min(date)) %>%
                        ungroup() %>%
                        dplyr::select(ID, pcode, season_approx, date) %>%
                        rename(cessation_date = date) %>%
                        filter(season_approx != '2020') # excluding rainy season 2020 for incomplete data
                      
            return(rainy_cessation)
            
          }













