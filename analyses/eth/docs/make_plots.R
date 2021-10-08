library(dplyr)
library(sf)
library(readxl)
library(ggplot2)
library(tidyr)
library(tmap)

# Read data ---------------------------------------------------------------

data_dir <- Sys.getenv("AA_DATA_DIR")
shapefile_path <- paste0(data_dir, "/public/raw/eth/cod_ab/ET_Admin_OCHA_2020")

shp_adm2 <- st_read(paste0(shapefile_path, "/eth_admbnda_adm2_csa_bofed_20201008.shp"))
shp_adm1 <- st_read(paste0(shapefile_path, "/eth_admbnda_adm1_csa_bofed_20201008.shp"))

processed_country_dir <- paste0(data_dir,"/public/processed/eth/")
exploration_country_dir <- paste0(data_dir,"/public/exploration/eth/")

emdat <- read_excel(paste0(exploration_country_dir, "emdat_public_2021_10_06_query_uid-vjop9c.xlsx"))


# Create plots ------------------------------------------------------------

plot_disaster_annual <- emdat %>%
  group_by(`Disaster Type`, Year) %>%
  summarise(n()) %>%
  mutate(Year = as.numeric(Year)) %>%
  complete(Year = 1965:2021) %>%
  mutate(`n()` = replace_na(`n()`, 0)) %>%
  ggplot(aes(x=Year, y=`n()`, group=`Disaster Type`, color =`Disaster Type`)) +
    geom_line() +
    theme_minimal() +
    labs(x='Year', y='Number') +
    theme(legend.position="bottom")



# Get number of events by admin 1 
# TODO: Fuzzy matching in case slight mismatch in spelling
# Join with events geolocated to adm2 as well

df_count <- data.frame(adm1 = shp_adm1$ADM1_EN)
df_count['num_flood'] <- 0
df_count['num_drought'] <- 0

emdat_flood <- emdat %>% filter(`Disaster Type` == 'Flood')
emdat_drought <- emdat %>% filter(`Disaster Type` == 'Drought')

# FLOOD
for (adm1 in shp_adm1$ADM1_EN) {
  for (event_loc in emdat_flood$`Geo Locations`){
    match <- grepl(adm1, event_loc, fixed=TRUE)
    if (match == TRUE){
      num_event = df_count[df_count$adm1 == adm1,]['num_flood'][[1]]
      df_count[df_count$adm1 == adm1,]['num_flood'] = num_event + 1
    }
  }
}

# DROUGHT
for (adm1 in shp_adm1$ADM1_EN) {
  for (event_loc in emdat_drought$`Geo Locations`){
    match <- grepl(adm1, event_loc, fixed=TRUE)
    if (match == TRUE){
      num_event = df_count[df_count$adm1 == adm1,]['num_flood'][[1]]
      df_count[df_count$adm1 == adm1,]['num_drought'] = num_event + 1
    }
  }
}

# Join to shapefile
shp_adm1 <- shp_adm1 %>%
  select(c(ADM1_EN, ADM1_PCODE)) %>%
  full_join(df_count, by=c('ADM1_EN' = 'adm1'))

flood_map <- tm_shape(shp_adm1) + 
  tm_fill(col = "num_flood", palette = "Blues", title = 'Flood events') +
  tm_borders(col = 'white') +
  tm_layout(frame = FALSE) +
  tm_text('ADM1_EN', size=0.75, col='black')

drought_map <- tm_shape(shp_adm1) + 
  tm_fill(col = "num_drought", title = 'Drought events') +
  tm_borders(col = 'white') +
  tm_layout(frame = FALSE) +
  tm_text('ADM1_EN', size=0.75, col='black')

drought_map
flood_map
