
computeLayerStat <- function(layer, stat, chirps_stat_values){
  
  # select 1 layer
  chirps_layer <- subset(chirps_masked, layer)
  
  # extract values from raster cells and compute stat
  chirps_layer.stat <- raster::extract(chirps_layer, mwi_adm2, fun = stat, df = T)
  
  # add daily stat to dataframe
  chirps_stat_values <- merge(chirps_stat_values, chirps_layer.stat, by = "ID", all.x = T)
  
  return(chirps_stat_values)
}

compute14dSum <- function(chirps_compiled.stat){
  
  # add pcodes to identify each polygon
  chirps_compiled.stat$pcode <- mwi_adm2$ADM2_PCODE
  
  # convert wide to long to get dates as rows
  chirps_compiled.stat_long <- gather(chirps_compiled.stat, date, total_prec, 2:(nbr_layers+1))
  
  # assign "zero" values to NA in total_prec
  chirps_compiled.stat_long$total_prec[is.na(chirps_compiled.stat_long$total_prec)] <- 0
  
  # reformat 'date' to a date format
  chirps_compiled.stat_long$date <- as.Date(chirps_compiled.stat_long$date, format = "X%Y.%m.%d")
  
  # convert to wide
  rolling_sum <-  chirps_compiled.stat_long %>%
    arrange(pcode, date) %>%
    group_by(pcode) %>%
    mutate(rollsum_14d = zoo::rollsum(total_prec, k = 14, fill = NA, align = 'right')
    ) 
  return(rolling_sum)
}
