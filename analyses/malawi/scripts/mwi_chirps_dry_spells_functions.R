
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

## TO DO: refactor to use convertToLongFormat()
compute14dSum <- function(data_compiled.stat){
  
  # add pcodes to identify each polygon
  data_compiled.stat$pcode <- mwi_adm2$ADM2_PCODE
  
  # convert wide to long to get dates as rows
  data_compiled.stat_long <- gather(data_compiled.stat, date, total_prec, 2:(nbr_layers+1))
  
  # assign "zero" values to NA in total_prec
  data_compiled.stat_long$total_prec[is.na(data_compiled.stat_long$total_prec)] <- 0
  
  # reformat 'date' to a date format
  data_compiled.stat_long$date <- as.Date(data_compiled.stat_long$date, format = "X%Y.%m.%d")
  
  # convert to wide
  rolling_sum <-  data_compiled.stat_long %>%
    arrange(pcode, date) %>%
    group_by(pcode) %>%
    mutate(rollsum_14d = zoo::rollsum(total_prec, k = 14, fill = NA, align = 'right')
    ) 
  return(rolling_sum)
}

## user-defined run-length encoding function in base R
runlengthEncoding <- function(x) {
     x <- rle(x)$lengths
     rep(seq_along(x), times=x)
}
