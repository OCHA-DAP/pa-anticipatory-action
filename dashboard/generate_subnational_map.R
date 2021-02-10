# global set up
options(scipen = 999) # remove scientific notation

# load libraries
library(tidyverse)
library(sf)
library(tmap)

# read in data to display
ipc_indices_data <- read.csv("data/foodinsecurity/ethiopia_foodinsec_trigger.csv") #Nov2020_FewsNetGlobalIPC.csv")
#last date of fewsnet and global ipc can differ, so select them separately and combine them
ipc_indices_data_latest_fn <- ipc_indices_data %>% filter(source == "FewsNet") %>% filter(date == max(date))
ipc_indices_data_latest_gbl <- ipc_indices_data %>% filter(source == "GlobalIPC") %>% filter(date == max(date))
ipc_indices_data_latest <- rbind(ipc_indices_data_latest_fn,ipc_indices_data_latest_gbl)

# import shapefiles
eth_adm1 <- st_read("data/shapefiles/ET_Admin1_OCHA_2019/eth_admbnda_adm1_csa_bofed_20190827.shp", stringsAsFactors = F)

# build datasets per country, source. Ensures all regions are represented if no projections in certain regions
# ETH FN
eth_fn <-  eth_adm1 %>%
             left_join(ipc_indices_data_latest_fn, by = c('ADM1_EN' = 'ADMIN1'))

# ETH GLB  
# For global IPC, it can be the case that for some dates, some admins do have ML1 projections but no ML2.
# In this case we want to show those admins as missing in the ML2 map
# Therefore, only select the admins with ML1/ML2 projections (indicated by a not-nan ML1/ML2 population)
eth_gbl_ml1 <-  eth_adm1 %>%
              left_join(ipc_indices_data_latest_gbl %>% drop_na(pop_ML1), by = c('ADM1_EN' = 'ADMIN1'))

eth_gbl_ml2 <-  eth_adm1 %>%
  left_join(ipc_indices_data_latest_gbl %>% drop_na(pop_ML2), by = c('ADM1_EN' = 'ADMIN1'))


# generate country-, source-, period-specific list of triggered regions
eth_fn_ML1_trigger_list <- eth_fn %>%
                            filter(threshold_reached_ML1 == 'True') 

eth_fn_ML2_trigger_list <- eth_fn %>%
                             filter(threshold_reached_ML2 == 'True') 

eth_gbl_ML1_trigger_list <- eth_gbl_ml1 %>%
                              filter(threshold_reached_ML1 == 'True') 

eth_gbl_ML2_trigger_list <- eth_gbl_ml2 %>%
                              filter(threshold_reached_ML2 == 'True') 

# produce country-, source-, period-specific maps
trigger_palette <- c("#e5e5e5", "#cc0000")

eth_fn_ML1_trigger_map <- eth_fn %>% 
                            tm_shape() +
                            tm_polygons("threshold_reached_ML1", 
                                        palette = trigger_palette, 
                                        title = "",
                                        textNA = "No data",) +
                            tm_text(text = "ADM1_EN", size = 0.75, col = "black") +
                            tm_format("NLD", title="Food Insecurity Criterion Reached", bg.color="white") +
                            tm_layout(legend.outside = TRUE) 

eth_fn_ML2_trigger_map <- eth_fn %>% 
                            tm_shape() +
                            tm_polygons("threshold_reached_ML2", 
                                        palette = trigger_palette, 
                                        title = "",
                                        textNA = "No data",) +
                            tm_text(text = "ADM1_EN", size = 0.75, col = "black") +
                            tm_format("NLD", title="Food Insecurity Criterion Reached", bg.color="white") +
                            tm_layout(legend.outside = TRUE) 


eth_gbl_ML1_trigger_map <- eth_gbl_ml1 %>% 
                              tm_shape() +
                              tm_polygons("threshold_reached_ML1", 
                                          palette = trigger_palette, 
                                          title = "",
                                          textNA = "No data",) +
                              tm_text(text = "ADM1_EN", size = 0.75, col = "black") +
                              tm_format("NLD", title="Food Insecurity Criterion Reached", bg.color="white") +
                              tm_layout(legend.outside = TRUE) 

eth_gbl_ML2_trigger_map <- eth_gbl_ml2 %>%
                              tm_shape() +
                              tm_polygons("threshold_reached_ML2", 
                                          palette = trigger_palette, 
                                          title = "",
                                          textNA = "No data",) +
                              tm_text(text = "ADM1_EN", size = 0.75, col = "black") +
                              tm_format("NLD", title="Food Insecurity Criterion Reached", bg.color="white") +
                              tm_layout(legend.outside = TRUE) 

  
  
  
  
  