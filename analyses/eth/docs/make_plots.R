library(tidyverse)
library(sf)
library(readxl)
library(tmap)
library(fuzzyjoin)

# Read data ---------------------------------------------------------------

data_dir <- Sys.getenv("AA_DATA_DIR")
shapefile_path <- file.path(data_dir, "public", "raw", "eth", "cod_ab", "ET_Admin_OCHA_2020")

shp_adm2 <- st_read(file.path(shapefile_path, "eth_admbnda_adm2_csa_bofed_20201008.shp"))
shp_adm1 <- st_read(file.path(shapefile_path, "eth_admbnda_adm1_csa_bofed_20201008.shp"))

processed_country_dir <- file.path(data_dir,"public", "processed", "eth")
exploration_country_dir <- file.path(data_dir,"public", "exploration", "eth")

emdat <- read_excel(file.path(exploration_country_dir, "emdat_public_2021_10_06_query_uid-vjop9c.xlsx"))


# Create plots ------------------------------------------------------------

plot_disaster_annual <- emdat %>%
  group_by(`Disaster Type`, Year) %>%
  summarise(n()) %>%
  mutate(Year = as.numeric(Year)) %>%
  complete(Year = 1965:2021, fill = list("n()" = 0))%>%
  ggplot(aes(x=Year, y=`n()`, group=`Disaster Type`, color =`Disaster Type`)) +
    geom_line() +
    theme_minimal() +
    labs(x='Year', y='Number') +
    theme(legend.position="bottom")

# Get number of events by admin 1 
# TODO: Fuzzy matching in case slight mismatch in spelling
# Join with events geolocated to adm2 as well
map_adm1 <- emdat %>%
  filter(!is.na(Location)) %>%
  fuzzyjoin::fuzzy_right_join(shp_adm1,
                              by = c(Location = "ADM1_EN"),
                              match_fun = ~ stringr::str_detect(tolower(.x), tolower(.y))) %>%
  st_as_sf() %>%
  group_by(ADM1_EN) %>%
  summarize(
    ADM1_PCODE = unique(ADM1_PCODE),
    num_flood = sum(`Disaster Type` == "Flood", na.rm = T),
    num_drought = sum(`Disaster Type` == "Drought", na.rm = T)
  )

flood_map <- tm_shape(map_adm1) + 
  tm_fill(col = "num_flood", palette = "Blues", title = 'Flood events') +
  tm_borders(col = 'white') +
  tm_layout(frame = FALSE) +
  tm_text('ADM1_EN', size=0.75, col='black')

drought_map <- tm_shape(map_adm1) + 
  tm_fill(col = "num_drought", title = 'Drought events') +
  tm_borders(col = 'white') +
  tm_layout(frame = FALSE) +
  tm_text('ADM1_EN', size=0.75, col='black')

# IDP data ----------------------------------------------------------------

idp_path <- file.path(exploration_country_dir, "hdx_dtm-ethiopia-site-assessment-round-26-dataset-june-july-2021.xlsx")
df_idp_cols <- as.character(read_excel(idp_path, n_max = 1, col_names = F))
df_idp <- read_excel(idp_path, skip = 2, col_names = df_idp_cols)

df_idp %>%
  group_by(`OCHA Zone P-Code`, `1.5.e.1: Reason for displacement`) %>%
  summarize(Count = n(), .groups = "drop") %>%
  pivot_wider(
    names_from = `1.5.e.1: Reason for displacement`,
    values_from = Count,
    values_fill = 0
  ) %>%
  readr::write_csv(file.path(exploration_country_dir, "idp_site_assessment_processed_reason_pcode.csv"))

