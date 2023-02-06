###### Analysis to understand if months in the upper tercile of the number of dry days overlap with months with dry spells
## Script for extracting the number of dry days in a month
######
# loading libraries
library(tidyverse)
library(sf)

######
# setting paths
data_dir <- Sys.getenv("AA_DATA_DIR")
shapefile_path <- paste0(data_dir, "/public/raw/mwi/cod_ab/mwi_adm_nso_20181016_shp")

######
# setting threshold to various values for dry days
dry_day_thresh <- c(0.3, 1, 2, 2.5, 3, 4)
sel_mon <- c("Jan", "Feb")
streakmon_start <- c("Jan", "Feb")
streakmon_end <- c("Jan", "Feb", "Mar")
n_dist <- 4

###### 
# reading in data and shapefiles
data_long_mean_values_2000_2021 <- readRDS(paste0(data_dir, "/public/processed/mwi/dry_spells/v1/data_long_mean_values_2000_2021.RDS"))
#data_values <- read_csv(paste0(data_dir, "/public/processed/mwi/dry_spells/v1/complete_list_per_pixel.csv"))
mwi_adm2 <- st_read(paste0(shapefile_path, "/mwi_admbnda_adm2_nso_20181016.shp"))

# getting adm2 pcodes in Southern region
southern_pcodes <- mwi_adm2$ADM2_PCODE[mwi_adm2$ADM1_EN == "Southern"]
# filtering the data for only Southern pcodes and testing if values are below thresholds
southern_data <- data_long_mean_values_2000_2021 %>%
    filter(pcode %in% southern_pcodes)# %>%
    #group_by(date) %>%
    #summarise(prec = mean(total_prec))

######
## function for computing upper tercile number of dry days in a month for all admin 2s.
dryday_count_fxn <- function(df, thresh){
    # summarise results by month
    # count number of dry days
    no_dry_days_mon <- df %>%
        mutate(val_less = if_else(total_prec <= thresh, 1, 0), 
               mon_yr = format(date,"%b_%Y"), mon = format(date,"%b")) %>%
        filter(mon %in% sel_mon) %>%
        group_by(pcode, mon_yr, mon) %>%
        summarise(no_dry_days = sum(val_less)) %>%
        ungroup() 
    
    # getting terciles
    terc_per_mon <- no_dry_days_mon %>%
        group_by(pcode, mon) %>%
        summarise(terciles = quantile(no_dry_days, 2/3))
    
    #write.csv(terc_per_mon, paste0("C:/Users/pauni/Desktop/Work/OCHA/Malawi/data/no_of_dry_days_terciles_per_month_", thresh,".csv"), row.names = F)
    # merging data frames
    merged_df <- merge(no_dry_days_mon, terc_per_mon, by = c("pcode", "mon"))
    # checking if values are in upper tercile
    df_extreme_mon <- merged_df %>%
        mutate(extreme = if_else(no_dry_days > terciles, 1, 0))
    # filtering only for those days
    df_extreme_dates <- df_extreme_mon %>%
        filter(extreme == 1)
    return(df_extreme_dates)
}
#dry_count1 <- dryday_count_fxn(df = southern_data, thresh = dry_day_thresh[1])
#dry_count2 <- dryday_count_fxn(df = southern_data, thresh = dry_day_thresh[2])
#dry_count3 <- dryday_count_fxn(df = southern_data, thresh = dry_day_thresh[3])
#dry_count4 <- dryday_count_fxn(df = southern_data, thresh = dry_day_thresh[4])
#dry_count5 <- dryday_count_fxn(df = southern_data, thresh = dry_day_thresh[5])
#dry_count6 <- dryday_count_fxn(df = southern_data, thresh = dry_day_thresh[6])
######
# presenting in a data frame
mon_df <- data.frame(matrix("", ncol = length(dry_day_thresh) * length(sel_mon), nrow = length(mon_val)))
col_count <- 0
for(i in 1:length(dry_day_thresh)){
    dry_count <- dryday_count_fxn(df = southern_data, thresh = dry_day_thresh[i])
    mon_val <- names(table(dry_count$mon_yr))[table(dry_count$mon_yr) > n_dist]
    for(j in 1:length(sel_mon)){
        col_count <- col_count + 1
        mon_col <- mon_val[startsWith(mon_val, sel_mon[j])]
        mon_df[1:length(mon_col), col_count] <- mon_col
    }
}
write.csv(mon_df, paste0(data_dir, "/public/processed/mwi/dry_spells/v1/jan_feb_review/no_dry_days_upper_tercile.csv"), row.names = F)
## check if month appears in more than n districts
#table(startsWith(unique(dry_count1$mon_yr), "Jan"))
#mon_val <- names(table(dry_count1$mon_yr))[table(dry_count1$mon_yr) > n_dist]
#mon_val
#table(startsWith(mon_val, "Jan"))

######
## function for computing upper tercile max dry days streaks per month for all admin 2s.
consecdays_max_fxn <- function(df, thresh){
    ### consecutive dry days
    df_consec <- df %>%
        group_by(pcode) %>%
        arrange(date) %>%
        mutate(val_less = if_else(total_prec <= thresh, 1, 0), 
               mon_yr = format(date,"%b_%Y"), mon = format(date,"%b"), 
               consec_dryday = sequence(rle(as.character(val_less))$lengths))
    
    ## check which ones were dry and started in Dec, Jan or Feb
    df_dry_consec <- df_consec %>%
        mutate(streak_start = date - consec_dryday + 1, streak_mon = format(streak_start,"%b")) %>%
        filter(val_less == 1 & streak_mon %in% streakmon_start & mon %in% streakmon_end) %>%
        group_by(pcode, streak_start) %>%
        filter(consec_dryday == max(consec_dryday)) %>%
        mutate(streak_end = streak_start + consec_dryday - 1,
               start_mon = format(streak_start,"%b_%Y")) 

    final_df <- df_dry_consec %>%
        group_by(pcode, start_mon) %>%
        filter(consec_dryday == max(consec_dryday)) %>%
        slice(1) %>%
        ungroup()

    # upper terciles values
    consec_mergeddf <- merge(final_df, (final_df %>% 
                                            #group_by(pcode, mon_yr, streak_mon) %>%
                                            #summarise(max_mon = max(consec_dryday)) %>%
                                            group_by(pcode, streak_mon) %>% 
                                            summarise(terciles = quantile(consec_dryday, 2/3))), by = c("pcode", "streak_mon")) %>%
        
        mutate(extreme = if_else(consec_dryday > terciles, 1, 0)) %>%
        filter(extreme == 1)
    
    return(consec_mergeddf)
}
#consec_count1 <- consecdays_max_fxn(df = southern_data, thresh = dry_day_thresh[1])
#consec_count2 <- consecdays_max_fxn(df = southern_data, thresh = dry_day_thresh[2])
#consec_count3 <- consecdays_max_fxn(df = southern_data, thresh = dry_day_thresh[3])
#consec_count4 <- consecdays_max_fxn(df = southern_data, thresh = dry_day_thresh[4])
#consec_count5 <- consecdays_max_fxn(df = southern_data, thresh = dry_day_thresh[5])
#consec_count6 <- consecdays_max_fxn(df = southern_data, thresh = dry_day_thresh[6])
#table(startsWith(unique(consec_count6$start_mon), "Jan"))
#mon_val <- names(table(consec_count6$start_mon))[table(consec_count6$start_mon) > n_dist]
#table(startsWith(mon_val, "Jan"))

######
# presenting in a data frame
consecmon_df <- data.frame(matrix("", ncol = length(dry_day_thresh) * length(sel_mon), nrow = length(mon_val)))
col_count <- 0
for(i in 1:length(dry_day_thresh)){
    consec_count <- consecdays_max_fxn(df = southern_data, thresh = dry_day_thresh[i])
    mon_val <- names(table(consec_count$start_mon))[table(consec_count$start_mon) > n_dist]
    for(j in 1:length(sel_mon)){
        col_count <- col_count + 1
        mon_col <- mon_val[startsWith(mon_val, sel_mon[j])]
        consecmon_df[1:length(mon_col), col_count] <- mon_col
    }
}
write.csv(consecmon_df, paste0(data_dir, "/public/processed/mwi/dry_spells/v1/jan_feb_review/consec_dry_days_upper_tercile_woMar.csv"), row.names = F)
