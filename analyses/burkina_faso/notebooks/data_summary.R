library(dplyr)
library(tmap)
library(ggplot2)
library(sf)
library(readxl)
library(stringr)
library(tidyverse)

data_dir <- Sys.getenv("AA_DATA_DIR")
bfa_dir <- paste0(data_dir, '/raw/burkina_faso/')

# UNIT OF ANALYSIS = ADM1 (Region)
# 13 regions in total

# 1. Load in and clean the data -------------------------------------------

# Drought
shp_drought <- st_read(paste0(bfa_dir, 'bfa_ica_droughtrisk_geonode_20180712/bfa_ica_droughtrisk_geonode_20180712.shp')) %>%
  mutate(adm1_name = tolower(adm1_name))

# Flood
shp_flood <- st_read(paste0(bfa_dir, 'bfa_ica_floodrisk_geonode_20180712/bfa_ica_floodrisk_geonode_20180712.shp')) %>%
  mutate(adm1_name = tolower(adm1_name))

# Population
shp_pop <- st_read(paste0(bfa_dir, 'bfa_ica_population_geonode_20180712/bfa_ica_population_geonode_20180712.shp')) %>%
  mutate(adm1_name = tolower(adm1_name))

# Population
df_pop <- read_excel(paste0(bfa_dir, 'bfa_population_statistic_2019_ign.xlsx')) %>% # Sheets for adm levels - first is for adm1
  slice(2:n()) %>%
  mutate(across(c('Fem': 'T_60plus'), as.numeric)) %>%
  mutate(adm1_name = tolower(ADM1_FR))#%>%
  #mutate(Total = Total/1000000)

# Poverty 
df_pov <- read_excel(paste0(bfa_dir, 'bfa-subnational-results-mpi-2020.xlsx')) %>% # Mult sheets - first is MPI by region
  select(c(7:13))%>%
  slice(9:n()) %>%
  slice(1:13) 
colnames(df_pov) <- c('adm1_name', 'mpi_adm0', 'mpi_adm1', 'hcr', 'dep_in', 'vuln_pov', 'sev_pov')
df_pov <- df_pov %>% 
  mutate(adm1_name = tolower(adm1_name)) %>%
  mutate(across(c('mpi_adm0':'sev_pov'), as.numeric))%>%
  mutate(adm1_name = ifelse(adm1_name=='boucle de mouhoun', 'boucle du mouhoun', adm1_name))

# --- IDP
df_idp <- read_excel(paste0(bfa_dir, 'situation_pdi_31122020.xlsx'))
colnames(df_idp) <- df_idp[6,]
df_idp <- df_idp %>% 
  slice(8:n()) %>%
  mutate(across(c('Hommes':'Nombre total de PDI'), as.numeric)) %>%
  mutate(adm1_name = tolower(Regions)) 

# Get the number of idps per adm1
df_idp_sum <- df_idp %>%
  group_by(adm1_name) %>%
  summarise(num_idp = sum(`Nombre total de PDI`)) %>%
  #mutate(num_idp = num_idp/1000000) %>%
  mutate(adm1_name = ifelse(adm1_name=='hauts bassins', 'hauts-bassins', ifelse(adm1_name=='plateaucentral', 'plateau central', adm1_name)))


# --- Food security 
df_fsec <- read_excel(paste0(bfa_dir, 'cadre_harmonise_caf_ipc.xlsx')) %>% # 1 sheet - not sure what the warnings are from...
  filter(adm0_pcod3 == 'BFA') %>%
  mutate(adm1_name = tolower(adm1_name)) %>%
  filter(exercise_year == 2020) %>%
  select(adm1_name, phase35, reference_label) %>%
  group_by(reference_label, adm1_name)%>%
  summarise(tot = sum(phase35)) %>%
  group_by(adm1_name) %>%
  summarise(ipc_plus_3_avg = mean(tot))%>%
  mutate(ipc_plus_3_avg = round(ipc_plus_3_avg, 0))
  #mutate(ipc_plus_3_avg = ipc_plus_3_avg/1000000)


# --- Operational presence 
df_op <- read_excel(paste0(bfa_dir, '3w-bfa-presence-operationnelle-310120211.xlsx')) %>% # 1 sheet
  mutate(adm1_name = tolower(Region))

# Get the number of activities per adm1
df_op_sum <- df_op %>% 
  group_by(adm1_name) %>%
  summarise(num_op = n())


# 2. Combine the variables ------------------------------------------------

adm1_name <- lapply(unique(df_fsec$adm1_name), tolower) # Convert to lower for better matching 

shp_sum <- df_idp_sum %>%
  full_join(df_op_sum, by='adm1_name') %>%
  full_join(shp_drought, by='adm1_name') %>%
  full_join(shp_flood , by='adm1_name') %>%
  full_join(df_pop, by='adm1_name') %>%
  full_join(df_pov, by='adm1_name') %>%
  full_join(df_fsec, by='adm1_name') %>%
  st_as_sf(sf_column_name = "geometry.x")%>%
  mutate(Dr_Text = as.factor(Dr_Text))%>%
  mutate(Dr_Text = factor(Dr_Text, levels=c('High', 'Medium', 'Low')))%>%
  mutate(FloodText = as.factor(FloodText))%>%
  mutate(FloodText = factor(FloodText, levels=c('High', 'Medium', 'Low')))

# 3. Basic maps -----------------------------------------------------------

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
    tm_credits(cap, position=c("right", "bottom"))
  return(m)
}

combine_map <- function(shp_sum){
  m_op <- make_map(shp_sum, 
                   'num_op', 
                   'BuGn', 
                   'Number of humanitarian\nactivities', 
                   'Source: OCHA, 2021')
  
  m_idp <- make_map(shp_sum, 
                    'num_idp', 
                    'Reds', 
                    'Number of IDPs',
                    'Source: OCHA, 2020')
  
  m_mpi <- make_map(shp_sum, 
                    'mpi_adm1', 
                    'PuRd', 
                    'Multidimensional\npoverty index',
                    'Source: Oxford Poverty and\nHuman Development Initiative, 2020')
  
  m_dr <- make_map(shp_sum, 
                   'Dr_Text', 
                   '-YlOrRd', 
                   'Drought risk', 
                   'Source: WFP, 1981-2015')
  
  m_fl <- make_map(shp_sum, 
                   'FloodText', 
                   '-Blues', 
                   'Flood risk', 
                   'Source: WFP, 2013')
  
  m_pop <- make_map(shp_sum, 
                    'Total', 
                    'Purples',
                    'Population',
                    'Source: OCHA, 2019')
  
  m_fsec <- make_map(shp_sum, 
                     'ipc_plus_3_avg',
                     'PuBuGn',
                     'Population\nIPC 3+',
                     'Source: FSNWG, 2020\nAveraged across year')
  
  w_title <- shp_sum %>%
    mutate(adm1_title = str_to_title(adm1_name))
  m_nm <- tm_shape(w_title) + tm_borders(alpha=0.5) +
    tm_text("adm1_title", size=0.5) +
    tm_layout(frame=FALSE)
  
  m_all <- tmap_arrange(m_nm, m_fl, m_op, m_dr, m_mpi, m_idp, m_pop, m_fsec, ncol=2)
  #tmap_save(m_all, 'map3.png', width=8, height=5, units='in', dpi=300)
  return(m_all)
}


# 4. Combine in table -----------------------------------------------------

df_sum <- shp_sum %>%
  select(c(adm1_name, num_op, num_idp, mpi_adm1, Dr_Text, FloodText, Total, ipc_plus_3_avg, geometry.x))
st_geometry(df_sum) <- NULL

# Clean up for display
df_sum_display <- df_sum %>%
  #mutate_if(is.numeric, round, 2)%>%
  mutate(adm1_name = str_to_title(adm1_name))%>%
  drop_na()


# 5. Population within hazard ---------------------------------------------



# Get pops within high drought risk
Drought <- shp_sum %>%
  filter(Dr_Text == 'High') %>%
  select(Total, M_60plus, F_60plus, num_idp)
st_geometry(Drought) <- NULL
Drought <- colSums(Drought)

# Get pops within high flood risk
Flood <- shp_sum %>%
  filter(FloodText == 'High') %>%
  select(Total, M_60plus, F_60plus, num_idp)
st_geometry(Flood) <- NULL
Flood <- colSums(Flood)

# Combine for comparison
high_hazard <- as.data.frame(rbind(Flood, Drought))

# Convert value to same units (single person)
#high_hazard <- high_hazard %>%
  #mutate(M_60plus = M_60plus/1000000) %>%
  #mutate(F_60plus = F_60plus/1000000) #%>%
  #mutate_if(is.numeric, round, 2)


# 6. Activities by sector -------------------------------------------------


plot_op_sum_type <- df_op %>%
  group_by(Cluster) %>%
  summarise(count = n()) %>%
  drop_na()%>%
  ggplot(aes(x=Cluster, y=count))+
  geom_bar(stat='identity') +
  theme_minimal()
