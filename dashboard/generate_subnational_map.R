# global set up
options(scipen = 999) # remove scientific notation

# load libraries
library(tidyverse)
library(sf)
library(tmap)

# read in data to display
# the csv should contain several columns, namely a source, date, ADMIN1, period_ML1, period_ML2, threshold_reached_ML1, threshold_reached_ML2
#and the percentage per ipc phase columns, i.e. perc_CS_3p, perc_CS_4, perc_ML1_3p, perc_ML1_4, perc_ML2_3p, perc_ML2_4
# the threshold_reached columns should have boolean values
ipc_indices_data <- read.csv("data/foodinsecurity/ethiopia_foodinsec_trigger.csv") 

# convert date string as a Date format
ipc_indices_data$date <- as.Date(ipc_indices_data$date, format = "%Y-%m-%d")

# select latest records. Last date of fewsnet and global ipc can differ so selected separately.
latest_report_per_source <- ipc_indices_data %>% 
                                group_by(source) %>%
                                slice(which.max(date)) %>% # keep only latest records for each source 
                                ungroup() %>%
                                select(source, date) %>%
                                unique()

ipc_indices_data_latest <- ipc_indices_data %>% 
                              right_join(latest_report_per_source, by = c('source' = 'source', 'date' = 'date'))

# import shapefiles
eth_adm1 <- st_read("data/shapefiles/ET_Admin_OCHA_2020/eth_admbnda_adm1_csa_bofed_20201008.shp", stringsAsFactors = F)

# build datasets per country, source. Ensures all regions are represented if no projections in certain regions
latest_fs <-  eth_adm1 %>%
               left_join(ipc_indices_data_latest, by = c('ADM1_EN' = 'ADMIN1'))

# For global IPC, it can be the case that for some dates, some admins do not have ML2 projections.
# In this case we want to show those admins as missing in the ML2 map
# Therefore, only select the admins with ML2 projections (indicated by a not-nan ML2 population)
#eth_gbl_ml2 <-  eth_adm1 %>%
 #                 left_join(ipc_indices_data_latest %>% drop_na(pop_ML2), by = c('ADM1_EN' = 'ADMIN1'))


# generate list of triggered regions across sources. Feb-May 2021 is ML2 for FewsNet and Jan-June 2021 is ML1 for GlobalIPC
fs_trigger_list <- latest_fs %>%
                  mutate(threshold_reached_H1_2021 = ifelse((source == 'FewsNet' & threshold_reached_ML2 == 'True') | (source == 'GlobalIPC' & threshold_reached_ML1 == 'True'), 1, 0)) %>%
                  group_by(ADM1_EN) %>%
                  mutate(threshold_reached_H1_2021 = ifelse(sum(threshold_reached_H1_2021) > 0, 1, 0)) %>% # assigns 1 to threshold_reached_H1_2021 if either source met threshold
                  ungroup() %>%
                  select(Shape_Leng, Shape_Area, ADM1_EN, threshold_reached_H1_2021, geometry) %>%
                  unique()

# produce  map of triggered regions across sources
trigger_palette <- c("#EEEEEE", "#F2645A") # grey first, tomato second

fs_trigger_map <- fs_trigger_list %>% 
                            tm_shape() +
                            tm_polygons("threshold_reached_H1_2021", 
                                        palette = trigger_palette, 
                                        title = "Food Insecurity Threshold Met",
                                        legend.show = FALSE) +
                            tm_text(text = "ADM1_EN", size = 0.75, col = "black") 
  
  
  