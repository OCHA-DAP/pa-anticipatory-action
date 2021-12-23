library(dplyr)
library(tmap)
library(ggplot2)
library(sf)
library(readxl)
library(stringr)
library(tidyverse)
library(ztable)

data_dir <- Sys.getenv("AA_DATA_DIR")
tcd_dir <- paste0(data_dir, '/raw/chad/risk/')

# UNIT OF ANALYSIS = ADM1. 23 adm1 regions. Note some datasets only available at adm2.

###############
#### 1. Load in and clean the data
###############

source("notebooks/data_import_cleaning.R")

###############
##### 2. Combine the variables 
###############

#adm1_name <- lapply(unique(df_fsec$adm1_name), tolower) # Convert to lower for better matching 

shp_sum <-  df_idp_sum %>% 
  full_join(df_op_sum, by = c('Admin1' = 'Region')) %>% 
  full_join(shp_drought, by = c('Admin1' = 'admin1Name_fr')) %>%
  dplyr::select(-Pcode1) %>%
  full_join(shp_flood, by = c('Admin1' = 'admin1Name_fr', 'admin1Pcode' = 'admin1Pcode')) %>%
  full_join(shp_shocks,  by = c('Admin1' = 'admin1Name_fr',  'admin1Pcode' = 'admin1Pcode')) %>% # exclude Pcode column before joining
  full_join(df_pov, by = c('Admin1' = 'adm1_name')) %>%
  full_join(df_fsec, by = c('Admin1' = 'adm1_name')) %>% # no data for N'Djamena
  full_join(df_pop, by = c('Admin1' = 'admin1Name_fr', 'admin1Pcode' = 'admin1Pcode')) %>%
  mutate(DroughtText = as.factor(DroughtText)) %>%
  mutate(DroughtText = factor(DroughtText, levels=c('High', 'Medium', 'Low'))) %>%
  mutate(FloodText = as.factor(FloodText)) %>%
  mutate(FloodText = factor(FloodText, levels=c('High', 'Medium', 'Low'))) %>%
  mutate(NSText = as.factor(NSText)) %>%
  mutate(NSText = factor(NSText, levels=c('High', 'Medium', 'Low'))) %>%
  full_join(shp[, c('admin1Name', 'geometry')], by = c('Admin1' = 'admin1Name')) %>%
  mutate(num_op = ifelse(is.na(num_op), 0, num_op)) %>% # replace NAs with zeroes
  st_as_sf(sf_column_name = "geometry") # transform df into an sf object

shp_sum[is.na(shp_sum$num_idp), "num_idp"] <- 0 # assumption: unlisted admin1's have no IDPs
shp_sum[is.na(shp_sum$num_op), "num_op"] <- 0 # assumption: unlisted admin1's have no operational presence

nrow(shp_sum) == nlevels(as.factor(adms$admin1Name_fr)) # check that all admin1's are represented

###############
# 3. Basic maps
###############

make_map <- function(shp, col, pal, title, cap){
  m <- tm_shape(shp)+
    tm_fill(col=col, palette=pal, title=title)+
    tm_shape(shp)+
    tm_borders('white', lwd=0.25)+
    tm_layout(frame=FALSE,
              legend.position = c('left', 'center'),
              legend.outside = TRUE,
              legend.outside.position = 'right',
              scale=0.85) +
    tm_credits(cap, position=c("right", "bottom"), size = 0.8)
  return(m)
}

combine_map <- function(shp_sum){
  m_op <- make_map(shp_sum, 
                   'num_op', 
                   'Purples', 
                   'Activités\nhumanitaires', 
                   'Source: OCHA, Juin 2020')
  
  m_idp <- make_map(shp_sum, 
                    'num_idp', 
                    'YlGnBu', 
                    'Personnes déplacées \ninternes',
                    'Source: OCHA, 2020')
  
  m_mpi <- make_map(shp_sum, 
                    'mpi_adm1', 
                    'Greens',
                    'Index multidimensionnel \nde la pauvreté',
                    'Source: OPHDI, 2020')
  
  m_dr <- make_map(shp_sum, 
                   'DroughtText', 
                   '-YlOrRd', 
                   'Risque de sécheresse', 
                   'Source: PAM, 1981-2015')
  
  m_fl <- make_map(shp_sum, 
                   'FloodText', 
                   '-Blues', 
                   'Risque d\'inondations', 
                   'Source: PAM, 2013')
  
  m_pop <- make_map(shp_sum, 
                    'Total_pop', 
                    'Greys',
                    'Population',
                    'Source: OCHA, 2020')
  
  m_fsec <- make_map(shp_sum, 
                     'ipc_plus_3_avg',
                     'OrRd',
                     'Pop. IPC 3+',
                     'Source: Cadre harmonisé, 2020')
  
  m_nat <- make_map(shp_sum, 
                     'NSText',
                     '-PuRd',
                     'Risque de chocs naturels',
                     'Source: PAM, 2017')
  
  m_all <- tmap_arrange(m_dr, m_fl, m_nat, m_pop, m_fsec, m_mpi, m_idp, m_op, ncol=3)
  
  #tmap_save(m_all, 'map3.png', width=8, height=5, units='in', dpi=300)
  return(m_all)
}

w_title <- shp_sum %>%
  mutate(adm1_title = str_to_title(Admin1))

m_nm <- tm_shape(w_title) + tm_borders(alpha=0.5) +
  tm_text("adm1_title", size=0.5) +
  tm_layout(frame=FALSE)
#tmap_save(m_nm, 'admins.png', width=8, height=5, units='in', dpi=300)

###############
# 4. Combine in table 
###############

df_sum <- shp_sum %>%
  dplyr::select(c(Admin1, DroughtText, FloodText, NSText, Total_pop, ipc_plus_3_avg, mpi_adm1, num_idp, num_op)) 

# Clean up for display
df_sum_display <- df_sum %>%
    mutate(Admin1 = str_to_title(Admin1)) # apply title case

df_sum_display$geometry <- NULL
