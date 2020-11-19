# global set up
options(scipen = 999) # remove scientific notation

# load libraries
library(tidyverse)
library(sf)
library(tmap)

# read in data to display
ipc_indices_data <- read.csv("Nov2020_FewsNetGlobalIPC.csv")

# import shapefiles
eth_adm1 <- st_read("shapefiles/ET_Admin1_OCHA_2019/eth_admbnda_adm1_csa_bofed_20190827.shp", stringsAsFactors = F)

# build datasets per country, source. Ensures all regions are represented if no projections in certain regions
# ETH FN
eth_gbl <-  eth_adm1 %>%
             left_join(ipc_indices_data[ipc_indices_data$Source == 'FewsNet',], by = c('ADM1_EN' = 'ADMIN1'))

# ETH GLB   
eth_gbl <-  eth_adm1 %>%
              left_join(ipc_indices_data[ipc_indices_data$Source == 'GlobalIPC',], by = c('ADM1_EN' = 'ADMIN1'))

# generate country-, source-, period-specific list of triggered regions
eth_fn_ML1_trigger_list <- eth_fn %>%
                            filter(trigger_ML1 == 'True') 

eth_fn_ML2_trigger_list <- eth_fn %>%
                             filter(trigger_ML2 == 'True') 

eth_gbl_ML1_trigger_list <- eth_gbl %>%
                              filter(trigger_ML1 == 'True') 

eth_gbl_ML2_trigger_list <- eth_gbl %>%
                              filter(trigger_ML2 == 'True') 

# produce country-, source-, period-specific maps
trigger_palette <- c("#e5e5e5", "#cc0000")

eth_fn_ML1_trigger_map <- eth_fn %>% 
                            tm_shape() +
                            tm_polygons("trigger_ML1", 
                                        palette = trigger_palette, 
                                        title = "") +
                            tm_text(text = "ADM1_EN", size = 0.75, col = "black") +
                            tm_format("NLD", title="Food Insecurity Criterion", bg.color="white") +
                            tm_layout(legend.outside = TRUE) 

eth_fn_ML2_trigger_map <- eth_fn %>% 
                            tm_shape() +
                            tm_polygons("trigger_ML2", 
                                        palette = trigger_palette, 
                                        title = "") +
                            tm_text(text = "ADM1_EN", size = 0.75, col = "black") +
                            tm_format("NLD", title="Food Insecurity Criterion", bg.color="white") +
                            tm_layout(legend.outside = TRUE) 


eth_gbl_ML1_trigger_map <- eth_gbl %>% 
                              tm_shape() +
                              tm_polygons("trigger_ML1", 
                                          palette = trigger_palette, 
                                          title = "") +
                              tm_text(text = "ADM1_EN", size = 0.75, col = "black") +
                              tm_format("NLD", title="Food Insecurity Criterion", bg.color="white") +
                              tm_layout(legend.outside = TRUE) 

eth_gbl_ML2_trigger_map <- eth_gbl %>% 
                              tm_shape() +
                              tm_polygons("trigger_ML2", 
                                          palette = trigger_palette, 
                                          title = "") +
                              tm_text(text = "ADM1_EN", size = 0.75, col = "black") +
                              tm_format("NLD", title="Food Insecurity Criterion", bg.color="white") +
                              tm_layout(legend.outside = TRUE) 

  
  
  
  
  