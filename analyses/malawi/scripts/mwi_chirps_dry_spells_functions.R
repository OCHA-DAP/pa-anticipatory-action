
computeLayerStat <- function(layer, stat, data_stat_values){
  
  # select 1 layer
  data_layer <- subset(data, layer)
  
  # extract values from raster cells and compute stat
  data_layer.stat <- raster::extract(data_layer, mwi_adm2, fun = stat, df = T)
  
  # add daily stat to dataframe
  data_stat_values <- merge(data_stat_values, data_layer.stat, by = "ID", all.x = T)
  
  return(data_stat_values)
}

# can create mwi_adm variable in computeLayerStat but would need to go back and fix all adm2 scripts so creating new function for adm3 for now
computeLayerStat_adm3 <- function(layer, stat, data_stat_values){
  
  # select 1 layer
  data_layer <- subset(data, layer)
  
  # extract values from raster cells and compute stat
  data_layer.stat <- raster::extract(data_layer, mwi_adm3, fun = stat, df = T)
  
  # add daily stat to dataframe
  data_stat_values <- merge(data_stat_values, data_layer.stat, by = "ID", all.x = T)
  
  return(data_stat_values)
}

# count number of pixels below threshold per adm2
computePixelsBelowThreshold <- function(layer, data_frame, threshold){
  
  # select 1 layer
  data_layer <- subset(data_all, layer)
  
  # extract values from raster cells and compute stat
  data_layer.stat <- raster::extract(data_layer, mwi_adm2, fun = function(x,...)sum(x <= threshold), df = T)
  
  # add daily stat to dataframe
  data_frame <- merge(data_frame, data_layer.stat, by = "ID", all.x = T)
  
  return(data_frame)
}

# count number of pixels in total and below threshold per adm2
computePixelsPerADM2 <- function(layer, data_frame){
  
  # select 1 layer
  data_layer <- subset(data_all, layer)
  
  # extract values from raster cells and compute stat
  data_layer.stat <- raster::extract(data_layer, mwi_adm2, fun = function(x,...)length(x), df = T)
  
  # add daily stat to dataframe
  data_frame <- merge(data_frame, data_layer.stat, by = "ID", all.x = T)
  
  return(data_frame)
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


# can create mwi_adm variable in computeLayerStat but would need to go back and fix all adm2 scripts so creating new function for adm3 for now
convertToLongFormatADM3 <- function(data.wideformat){
  
  # add pcodes to identify each polygon
  data.wideformat$pcode <- mwi_adm3$ADM3_PCODE
  
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

## compute rolling sum
computeRollingSumPerPixel <- function(dataframe_long, window){
  
  rolling_sum <-  dataframe_long %>%
    arrange(cell, date) %>%
    group_by(cell) %>%
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
  data_long$min_cum_40mm_bin <- ifelse(data_long$rollsum_10d >= 40, 1, 0) # is this day in a 40+mm period?
  
  # identify 10-day dry spells (10 consecutive days with less than 2mm of total rain) yearound
  data_long$less_than_cum_2mm_bin <- ifelse(data_long$rollsum_10d < 2, 1, 0) # is this day in a dry period (<2mm)?
  
  # verify no 10-day dry spells following in 30 days of first day with >=40mm cum sum yearound
  data_long <- data_long %>% 
                            group_by(pcode) %>%
                            mutate(nbr_dry_spell_days_win_30d = zoo::rollsum(less_than_cum_2mm_bin, k = 30, fill = NA, align = 'left'),
                                   followed_by_ds_win_30d_bin = ifelse(nbr_dry_spell_days_win_30d > 0, 1, 0))  
                          
  # select earliest date per season_approx after 1 Nov that meets both criteria
  rainy_onsets <- data_long %>%
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

## compute onset date for every rainy season per pixel
findRainyOnsetPerPixel <- function() {
  
  # identify 10-day periods with at least 40mm cumulative yearound
  data_long$min_cum_40mm_bin <- ifelse(data_long$rollsum_10d >= 40, 1, 0) # is this day in a 40+mm period?
 
  # identify 10-day dry spells (10 consecutive days with less than 2mm of total rain) yearound
  data_long$less_than_cum_2mm_bin <- ifelse(data_long$rollsum_10d < 2, 1, 0) # is this day in a dry period (<2mm)?
  
  # verify no 10-day dry spells following in 30 days of first day with >=40mm cum sum yearound
  data_long <- data_long %>% 
    group_by(cell) %>%
    mutate(nbr_dry_spell_days_win_30d = zoo::rollsum(less_than_cum_2mm_bin, k = 30, fill = NA, align = 'left'),
           followed_by_ds_win_30d_bin = ifelse(nbr_dry_spell_days_win_30d > 0, 1, 0))  
  
  # select earliest date per season_approx after 1 Nov that meets both criteria
  rainy_onsets <- data_long %>%
    filter(season_approx != 'outside rainy season') %>% # exclude Aug-Sept
    mutate(meets_onset_criteria = ifelse(min_cum_40mm_bin == 1 & followed_by_ds_win_30d_bin == 0, 1, 0)) %>% # min 40mm in 15 days and not followed by 10d dry spells within 30 days
    filter(meets_onset_criteria == 1 & month %in% c(11, 12, 1, 2)) %>% # period post 1 Nov. Allows Nov-Feb for onsets)
    group_by(cell, season_approx) %>%
    slice(which.min(date)) %>%
    ungroup() %>%
    dplyr::select(cell, season_approx, date) %>%
    rename(onset_date = date) 
  
  rainy_onsets$onset_date[rainy_onsets$season_approx == '1999'] <- NA # set values for 1999 season as NA since no data Oct-Dec 1999 available
  
  return(rainy_onsets)
  
}


## compute cessation date for every rainy season per adm2 NOTE: uses the left alignment / "back" method to compute the 15d rolling sum
findRainyCessation <- function() {
  
   # identify 15-day periods of up to 25mm cum
    data_long$max_cum_25mm_bin <- ifelse(data_long$rollsum_15d_back <= 25, 1, 0) # is this day in a 25mm or less 15d period?
  
    # select earliest date per season_approx after 15 March that meets criterion
    rainy_cessation <- data_long %>%
                          filter((month >= 4 & month < 8) | (month == 3 & day >= 15)) %>% # in Mar on or after the 15th, or between April and Aug exclusively
                          filter(max_cum_25mm_bin == 1) %>% # meet criterion for cessation (= 25mm or less cumulative rainfall over 15 days)
                          group_by(pcode, season_approx) %>%
                          slice(which.min(date)) %>%
                          ungroup() %>%
                          dplyr::select(ID, pcode, season_approx, date) %>%
                          rename(cessation_date = date) 
    
    return(rainy_cessation)

}


## compute cessation date for every rainy season per pixel. Uses right alignment to compute the 15-d rolling sum. includes the dry spell that ends the rainy season in the rainy season
findRainyCessation_original <- function() {
  
  # identify 15-day periods of up to 25mm cum
  data_long$max_cum_25mm_bin <- ifelse(data_long$rollsum_15d <= 25, 1, 0) # is this day in a 25mm or less 15d period?
  
    # select earliest date per season_approx after 15 March that meets criterion
  rainy_cessation <- data_long %>%
                        filter((month >= 4 & month < 8) | (month == 3 & day >= 15)) %>% # in Mar on or after the 15th, or between April and Aug exclusively
                        filter(max_cum_25mm_bin == 1) %>% # meet criterion for cessation (= 25mm or less cumulative rainfall over 15 days)
                        group_by(cell, season_approx) %>%
                        slice(which.min(date)) %>%
                        ungroup() %>%
                        dplyr::select(cell, season_approx, date) %>%
                        rename(cessation_date = date) 
  
  return(rainy_cessation)
  
}

# create binary for days in a dry spell (14-d <=2mm cum) per pixel
listDSDaysPerPixel <- function(i) {
               
    # take cell number
    cell_number <- dry_spells_during_rainy_season_list$cell[i]
    
    # take season_approx
    season_approx_value <- dry_spells_during_rainy_season_list$season_approx[i]
    
    # take adm2 
    adm2_name <- dry_spells_during_rainy_season_list$ADM2_EN[i]
    
    # take region
    adm1_name <- dry_spells_during_rainy_season_list$region[i]
    
    # generate list of dates of the dry spell
    dates_list <- data.frame(date = seq(from = dry_spells_during_rainy_season_list$dry_spell_first_date[i], 
                      to = dry_spells_during_rainy_season_list$dry_spell_last_date[i], 
                      by = 1))
    # add cell number
    dates_list$cell <- cell_number
    
    # add adm names
    dates_list$region <- adm1_name
    dates_list$ADM2_EN <- adm2_name
    
    # add season_approx
    dates_list$season_approx <- season_approx_value
    
    return(dates_list)
  }


## user-defined run-length encoding function in base R
runlengthEncoding <- function(x) {
  x <- rle(x)$lengths
  rep(seq_along(x), times=x)
}











