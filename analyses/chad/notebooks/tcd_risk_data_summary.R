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

source("data_import_cleaning.R")

###############
##### 2. Combine the variables 
###############

#adm1_name <- lapply(unique(df_fsec$adm1_name), tolower) # Convert to lower for better matching 

shp_sum <- df_idp_sum %>% # verified the 1:1 relationship between pcode and name at adm1
  full_join(df_op_sum, by = c('Admin1' = 'Region')) %>% 
  full_join(shp_drought, by = c('Admin1' = 'admin1Name_fr')) %>%
  dplyr::select(-Pcode1) %>%
  full_join(shp_flood, by = c('Admin1' = 'admin1Name_fr', 'admin1Pcode' = 'admin1Pcode')) %>%
  full_join(df_pop, by = c('Admin1' = 'admin1Name_fr', 'admin1Pcode' = 'admin1Pcode')) %>%
  full_join(df_pov, by = c('Admin1' = 'adm1_name')) %>%
  full_join(df_fsec, by = c('Admin1' = 'adm1_name')) %>%
  mutate(DroughtText = as.factor(DroughtText)) %>%
  mutate(DroughtText = factor(DroughtText, levels=c('High', 'Medium', 'Low'))) %>%
  mutate(FloodText = as.factor(FloodText)) %>%
  mutate(FloodText = factor(FloodText, levels=c('High', 'Medium', 'Low'))) %>%
  full_join(shp[, c('admin1Name', 'geometry')], by = c('Admin1' = 'admin1Name')) %>%
  mutate(num_op = ifelse(is.na(num_op), 0, num_op)) %>% # replace NAs with zeroes
  st_as_sf(sf_column_name = "geometry") # transform df into an sf object

###############
# 3. Basic maps
###############

make_map <- function(shp, col, pal, title, cap){
  m <- tm_shape(shp)+
    tm_fill(col=col, palette=pal, title=title)+
    tm_shape(shp)+
    tm_borders('white', lwd=0.25)+
    tm_layout(frame=FALSE,
              legend.position = c('left', 'top'),
              legend.outside = TRUE,
              legend.outside.position = 'right',
              scale=0.75) +
    tm_credits(cap, position=c("centre", "bottom"))
  return(m)
}

combine_map <- function(shp_sum){
  m_op <- make_map(shp_sum, 
                   'num_op', 
                   'BuGn', 
                   'Nombre d\'activités\nhumanitaires', 
                   'Source: OCHA, June 2020')
  
  m_idp <- make_map(shp_sum, 
                    'num_idp', 
                    'Reds', 
                    'Personnes déplacées internes',
                    'Source: OCHA, 2020')
  
  m_mpi <- make_map(shp_sum, 
                    'mpi_adm1', 
                    #'PuRd', 
                    'BuPu',
                    'Index multidimensionnel \nde la pauvreté',
                    'Source: Oxford Poverty and\nHuman Development Initiative, 2020')
  
  m_dr <- make_map(shp_sum, 
                   'DroughtText', 
                   '-YlOrRd', 
                   'Risque de sécheresse', 
                   'Source: WFP, 1981-2015')
  
  m_fl <- make_map(shp_sum, 
                   'FloodText', 
                   '-Blues', 
                   'Risque d\'inondations', 
                   'Source: WFP, 2013')
  
  m_pop <- make_map(shp_sum, 
                    'Total_pop', 
                    'Purples',
                    'Population',
                    'Source: OCHA, 2020')
  
  m_fsec <- make_map(shp_sum, 
                     'ipc_plus_3_avg',
                     'PuBuGn',
                     'Population\nIPC 3+',
                     'Source: Cadre harmonisé, 2020\nMoyenne sur l\'année')
  
  w_title <- shp_sum %>%
    mutate(adm1_title = str_to_title(Admin1))
    m_nm <- tm_shape(w_title) + tm_borders(alpha=0.5) +
    tm_text("adm1_title", size=0.5) +
    tm_layout(frame=FALSE)
  
  m_all <- tmap_arrange(m_nm, m_fl, m_op, m_dr, m_mpi, m_idp, m_pop, m_fsec, ncol=2)
  #tmap_save(m_all, 'map3.png', width=8, height=5, units='in', dpi=300)
  return(m_all)
}

###############
# 4. Combine in table 
###############

df_sum <- shp_sum %>%
  dplyr::select(c(Admin1, num_op, num_idp, mpi_adm1, DroughtText, FloodText, Total_pop, ipc_plus_3_avg))

# Clean up for display
df_sum_display <- df_sum %>%
  #mutate_if(is.numeric, round, 2)%>%
  mutate(Admin1 = str_to_title(Admin1)) %>% # apply title case
  drop_na()

###############
# 5. Population within hazard 
###############





# 
# 
# # Get pops within high drought risk
# Drought <- shp_sum %>%
#   filter(DroughtText == 'High') %>%
#   select(Total_pop, M_60plus, F_60plus, num_idp)
# 
# st_geometry(Drought) <- NULL
# Drought <- colSums(Drought)
# 
# # Get pops within high flood risk
# Flood <- shp_sum %>%
#   filter(FloodText == 'High') %>%
#   select(Total, M_60plus, F_60plus, num_idp)
# st_geometry(Flood) <- NULL
# Flood <- colSums(Flood)
# 
# # Combine for comparison
# high_hazard <- as.data.frame(rbind(Flood, Drought))

# Convert value to same units (single person)
#high_hazard <- high_hazard %>%
  #mutate(M_60plus = M_60plus/1000000) %>%
  #mutate(F_60plus = F_60plus/1000000) #%>%
  #mutate_if(is.numeric, round, 2)

