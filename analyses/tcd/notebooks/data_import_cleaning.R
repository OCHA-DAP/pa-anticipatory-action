library(dplyr)
tcd_dir <- paste0(data_dir, '/public/raw/tcd')

## Load in and transform datasets

# list of adms regions (70 Départements (Admin2) and 23 Regions (Admin1), 1:1 relationship between pcodes and names)
cod_ab_path <- paste0(data_dir, '/public/raw/tcd/cod_ab', '/tcd_adminboundaries_tabulardata-20170616.xlsx')
adms <- read_excel(cod_ab_path, sheet = 2) %>%
  dplyr::select(admin2Pcode, admin2Name_fr, admin1Pcode, admin1Name_fr)

# Shapefiles 
shapefile_path <- paste0(data_dir, '/public/raw/tcd/cod_ab', '/tcd_admbnda_adm1_ocha/tcd_admbnda_adm1_ocha.shp')
shp <- st_read(shapefile_path)

# Drought (data at adm2) # DO NOT USE 'adm2_code'. NPGS = number of poor growing seasons
drought_risk_filepath <- public/raw/tcd/risk/tcd_ica_droughtrisk_geonode_mar2017/tcd_ica_droughtrisk_geonode_mar2017.shp


shp_drought_all <- st_read(paste0(tcd_dir, 'tcd_ica_droughtrisk_geonode_mar2017/tcd_ica_droughtrisk_geonode_mar2017.shp')) # Iriba listed but is a town not an adm2. It is in Wadi Hawar (adm2)

shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Tibesti-Ouest"] <- 'Tibesti Ouest' # name changes don't work in a user-defined function
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Megri"] <- 'Mégri'
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Barh El Gazel Nord"] <- 'Barh-El-Gazel Nord'
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Guera"] <- 'Guéra'
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Wadi-Bissam"] <- 'Wadi Bissam'
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Gueni"] <- 'Guéni'
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Ngourkoussou"] <- 'Ngourkosso'
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Dodje"] <- 'Dodjé'
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Barh-Azoum"] <- 'Bahr-Azoum'
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Aboudeia"] <- 'Aboudéia'
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Haraze Mangueigne"] <- 'Haraze-Mangueigne'
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Batha-Est"] <- 'Batha Est'
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Batha-Ouest"] <- 'Batha Ouest'
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Kouh-Ouest"] <- 'Kouh Ouest'
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Kouh-Est"] <- 'Kouh Est'
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Barh El Gazel Sud"] <- 'Barh-El-Gazel Sud'
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Mayo Boneye"] <- 'Mayo-Boneye'
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Ndjamena"] <- "N'Djaména"
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Nord-Kanem"] <- 'Nord Kanem'
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Barh El Gazel Ouest"] <- 'Barh-El-Gazel Ouest'
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Kabia"] <- 'Kabbia'
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Lac-Léré"] <- 'Lac Léré'
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Mayo Binder"] <- 'Mayo-Binder'
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Djourouf Al Ahmar"] <- 'Djourf Al Ahmar'
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Barh-Kôh"] <- 'Bahr-Köh'
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Tandjile-Centre"] <- 'Tandjilé Centre'
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Tandjile-Est"] <- 'Tandjilé Est'
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Tandjile-Ouest"] <- 'Tandjile Ouest'
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Assongha"] <- 'Assoungha'
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Mayo-Lemi"] <- 'Mayo-Lemié'
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Wardi Hawar"] <- 'Wadi Hawar'
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Amdjarass"] <- 'Am-Djarass'
shp_drought_all$adm2_name[shp_drought_all$adm2_name == "Tibesti-Est"] <- 'Tibesti Est'

shp_drought_all <- shp_drought_all %>% right_join(adms, by = c('adm2_name' = 'admin2Name_fr')) 

st_geometry(shp_drought_all) <- NULL # remove geometry

shp_drought <- shp_drought_all %>% # compute mean drought index per admin1 (=take mean of NPGS, then assign it a class and Dr_text)
  dplyr::select(admin1Pcode, admin1Name_fr, NPGS) %>% 
  group_by(admin1Pcode, admin1Name_fr) %>% 
  summarise(mean_NPGS = mean(NPGS, na.rm = T)) %>% # removes NA and may inflate average if NAs were true zeroes
  mutate(DroughtClass = ifelse(mean_NPGS <= 6, 1, 
                             ifelse(mean_NPGS > 6 & mean_NPGS <= 11, 2,
                                    ifelse(mean_NPGS > 11, 3, NA))),
         DroughtText = ifelse(DroughtClass == 1, "Low",
                          ifelse(DroughtClass == 2, "Medium", 
                                 ifelse(DroughtClass == 3, "High", NA))))

# Flood
shp_flood_all <- st_read(paste0(tcd_dir, 'tcd_ica_floodrisk_geonode_mar2017/tcd_ica_floodrisk_geonode_mar2017.shp')) # Iriba listed but is a town not an adm2. It is in Wadi Hawar (adm2)

shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Tibesti-Ouest"] <- 'Tibesti Ouest'
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Megri"] <- 'Mégri'
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Barh El Gazel Nord"] <- 'Barh-El-Gazel Nord'
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Guera"] <- 'Guéra'
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Wadi-Bissam"] <- 'Wadi Bissam'
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Gueni"] <- 'Guéni'
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Ngourkoussou"] <- 'Ngourkosso'
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Dodje"] <- 'Dodjé'
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Barh-Azoum"] <- 'Bahr-Azoum'
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Aboudeia"] <- 'Aboudéia'
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Haraze Mangueigne"] <- 'Haraze-Mangueigne'
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Batha-Est"] <- 'Batha Est'
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Batha-Ouest"] <- 'Batha Ouest'
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Kouh-Ouest"] <- 'Kouh Ouest'
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Kouh-Est"] <- 'Kouh Est'
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Barh El Gazel Sud"] <- 'Barh-El-Gazel Sud'
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Mayo Boneye"] <- 'Mayo-Boneye'
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Ndjamena"] <- "N'Djaména"
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Nord-Kanem"] <- 'Nord Kanem'
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Barh El Gazel Ouest"] <- 'Barh-El-Gazel Ouest'
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Kabia"] <- 'Kabbia'
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Lac-Léré"] <- 'Lac Léré'
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Mayo Binder"] <- 'Mayo-Binder'
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Djourouf Al Ahmar"] <- 'Djourf Al Ahmar'
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Barh-Kôh"] <- 'Bahr-Köh'
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Tandjile-Centre"] <- 'Tandjilé Centre'
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Tandjile-Est"] <- 'Tandjilé Est'
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Tandjile-Ouest"] <- 'Tandjile Ouest'
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Assongha"] <- 'Assoungha'
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Mayo-Lemi"] <- 'Mayo-Lemié'
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Wardi Hawar"] <- 'Wadi Hawar'
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Amdjarass"] <- 'Am-Djarass'
shp_flood_all$adm2_name[shp_flood_all$adm2_name == "Tibesti-Est"] <- 'Tibesti Est'

shp_flood_all <- shp_flood_all %>% right_join(adms, by = c('adm2_name' = 'admin2Name_fr')) 

st_geometry(shp_flood_all) <- NULL # remove geometry

shp_flood <- shp_flood_all %>% # compute mean drought index per admin1 (=take mean of NPGS, then assign it a class and Dr_text)
  dplyr::select(admin1Pcode, admin1Name_fr, FloodRisk) %>% 
  group_by(admin1Pcode, admin1Name_fr) %>% 
  summarise(mean_FloodRisk = round(mean(FloodRisk, na.rm = T), 0)) %>% # removes NA and may inflate average if NAs were true zeroes. Rounding, maybe should be ceiling/floor function instead
  mutate(FloodClass = ifelse(mean_FloodRisk <= 2, 1, 
                             ifelse(mean_FloodRisk >= 3 & mean_FloodRisk <= 4, 2,
                                    ifelse(mean_FloodRisk >= 5, 3, NA))),
         FloodText = ifelse(FloodClass == 1, "Low",
                          ifelse(FloodClass == 2, "Medium", 
                                 ifelse(FloodClass == 3, "High", NA))))

# Natural shock risk
shp_shocks_all <- st_read(paste0(tcd_dir, 'tcd_ica_naturalshocksrisk_geonode_mar2017/tcd_ica_naturalshocksrisk_geonode_mar2017.shp'))

shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Tibesti-Ouest"] <- 'Tibesti Ouest'
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Megri"] <- 'Mégri'
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Barh El Gazel Nord"] <- 'Barh-El-Gazel Nord'
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Guera"] <- 'Guéra'
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Wadi-Bissam"] <- 'Wadi Bissam'
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Gueni"] <- 'Guéni'
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Ngourkoussou"] <- 'Ngourkosso'
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Dodje"] <- 'Dodjé'
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Barh-Azoum"] <- 'Bahr-Azoum'
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Aboudeia"] <- 'Aboudéia'
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Haraze Mangueigne"] <- 'Haraze-Mangueigne'
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Batha-Est"] <- 'Batha Est'
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Batha-Ouest"] <- 'Batha Ouest'
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Kouh-Ouest"] <- 'Kouh Ouest'
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Kouh-Est"] <- 'Kouh Est'
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Barh El Gazel Sud"] <- 'Barh-El-Gazel Sud'
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Mayo Boneye"] <- 'Mayo-Boneye'
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Ndjamena"] <- "N'Djaména"
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Nord-Kanem"] <- 'Nord Kanem'
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Barh El Gazel Ouest"] <- 'Barh-El-Gazel Ouest'
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Kabia"] <- 'Kabbia'
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Lac-Léré"] <- 'Lac Léré'
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Mayo Binder"] <- 'Mayo-Binder'
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Djourouf Al Ahmar"] <- 'Djourf Al Ahmar'
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Barh-Kôh"] <- 'Bahr-Köh'
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Tandjile-Centre"] <- 'Tandjilé Centre'
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Tandjile-Est"] <- 'Tandjilé Est'
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Tandjile-Ouest"] <- 'Tandjile Ouest'
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Assongha"] <- 'Assoungha'
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Mayo-Lemi"] <- 'Mayo-Lemié'
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Wardi Hawar"] <- 'Wadi Hawar'
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Amdjarass"] <- 'Am-Djarass'
shp_shocks_all$adm2_name[shp_shocks_all$adm2_name == "Tibesti-Est"] <- 'Tibesti Est'

shp_shocks_all <- shp_shocks_all %>% right_join(adms, by = c('adm2_name' = 'admin2Name_fr')) 

st_geometry(shp_shocks_all) <- NULL # remove geometry

shp_shocks <- shp_shocks_all %>% # compute mean drought index per admin1 (= take mean of NPGS, then assign it a class and Dr_text)
  dplyr::select(admin1Pcode, admin1Name_fr, NS_Risk) %>% 
  group_by(admin1Pcode, admin1Name_fr) %>% 
  summarise(mean_NS_Risk = round(mean(NS_Risk, na.rm = T), 0)) %>% # removes NA and may inflate average if NAs were true zeroes. Rounding, maybe should be ceiling/floor function instead
  mutate(NSClass = ifelse(mean_NS_Risk <= 2, 1, 
                             ifelse(mean_NS_Risk >= 3 & mean_NS_Risk < 4, 2,
                                    ifelse(mean_NS_Risk >= 4, 3, NA))),
         NSText = ifelse(NSClass == 1, "Low",
                          ifelse(NSClass == 2, "Medium", 
                                 ifelse(NSClass == 3, "High", NA))))

# ICA Categories
shp_ica_all <- st_read(paste0(tcd_dir, 'tcd_ica_categories_areas_geonode_mar2017/tcd_ica_categories_areas_geonode_mar2017.shp'))

shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Tibesti-Ouest"] <- 'Tibesti Ouest'
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Megri"] <- 'Mégri'
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Barh El Gazel Nord"] <- 'Barh-El-Gazel Nord'
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Guera"] <- 'Guéra'
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Wadi-Bissam"] <- 'Wadi Bissam'
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Gueni"] <- 'Guéni'
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Ngourkoussou"] <- 'Ngourkosso'
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Dodje"] <- 'Dodjé'
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Barh-Azoum"] <- 'Bahr-Azoum'
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Aboudeia"] <- 'Aboudéia'
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Haraze Mangueigne"] <- 'Haraze-Mangueigne'
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Batha-Est"] <- 'Batha Est'
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Batha-Ouest"] <- 'Batha Ouest'
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Kouh-Ouest"] <- 'Kouh Ouest'
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Kouh-Est"] <- 'Kouh Est'
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Barh El Gazel Sud"] <- 'Barh-El-Gazel Sud'
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Mayo Boneye"] <- 'Mayo-Boneye'
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Ndjamena"] <- "N'Djaména"
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Nord-Kanem"] <- 'Nord Kanem'
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Barh El Gazel Ouest"] <- 'Barh-El-Gazel Ouest'
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Kabia"] <- 'Kabbia'
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Lac-Léré"] <- 'Lac Léré'
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Mayo Binder"] <- 'Mayo-Binder'
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Djourouf Al Ahmar"] <- 'Djourf Al Ahmar'
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Barh-Kôh"] <- 'Bahr-Köh'
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Tandjile-Centre"] <- 'Tandjilé Centre'
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Tandjile-Est"] <- 'Tandjilé Est'
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Tandjile-Ouest"] <- 'Tandjile Ouest'
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Assongha"] <- 'Assoungha'
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Mayo-Lemi"] <- 'Mayo-Lemié'
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Wardi Hawar"] <- 'Wadi Hawar'
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Amdjarass"] <- 'Am-Djarass'
shp_ica_all$adm2_name[shp_ica_all$adm2_name == "Tibesti-Est"] <- 'Tibesti Est'

shp_ica_all <- shp_ica_all %>% right_join(adms, by = c('adm2_name' = 'admin2Name_fr')) 

st_geometry(shp_ica_all) <- NULL # remove geometry


# Population
#df_pop <- read_excel(paste0(tcd_dir, 'tcd_admpop_2020.xlsx'), sheet = 2) %>% # adm 0-2 | 2021 projected sex and age disaggregated population statistics
df_pop <- read_excel(paste0(tcd_dir, 'tcd_admpop_2019.xlsx'), sheet = 2) %>% # adm 0-2 disaggregated by gender and age 
  mutate(across(c('F': 'T_80plus'), as.numeric)) %>% # reformat into numeric 
  rename(Female = 'F',
         Male = 'M',
         Total_pop = 'T')

# IDP
df_idp <- read_excel(paste0(tcd_dir, 'tcd_data_cod_ps_idp_rt_rf_20201130.xlsx'), skip = 3)

df_idp$Admin2[df_idp$Admin2 == "Bahr-Koh"] <- 'Bahr-Köh'
df_idp$Admin1[df_idp$Admin1 == "Ouaddai"] <- 'Ouaddaï'

df_idp_sum <- df_idp %>% # Get the number of idps per adm1
  group_by(Admin1) %>%
  summarise(num_idp = sum(`NbPersonnes`))

# Food security 
df_fsec_all <- read_excel(paste0(tcd_dir, 'cadre_harmonise_caf_ipc.xlsx')) # 1 sheet 

df_fsec_all$adm1_name[df_fsec_all$adm1_name == "Ouaddai"] <- 'Ouaddaï'
df_fsec_all$adm1_name[df_fsec_all$adm1_name == "Ennedi-Ouest"] <- 'Ennedi Ouest'
df_fsec_all$adm1_name[df_fsec_all$adm1_name == "Guera"] <- 'Guéra'
df_fsec_all$adm1_name[df_fsec_all$adm1_name == "Mayo Kebbi Est"] <- 'Mayo-Kebbi Est'
df_fsec_all$adm1_name[df_fsec_all$adm1_name == "Ennedi-Est"] <- 'Ennedi Est'
df_fsec_all$adm1_name[df_fsec_all$adm1_name == "Tandjile"] <- 'Tandjilé'

df_fsec <- df_fsec_all %>%
  filter(adm0_pcod3 == 'TCD' & exercise_year == 2020) %>% # keep most recent full year
  dplyr::select(adm1_pcod3, adm1_name, phase35, reference_label) %>% # keep number in IPC Phase 3 to 5
  group_by(reference_label, adm1_pcod3, adm1_name) %>%
  summarise(tot = sum(phase35)) %>%
  ungroup() %>%
  group_by(adm1_name) %>%
  summarise(ipc_plus_3_avg = mean(tot)) %>% # compute average over the year
  mutate(ipc_plus_3_avg = round(ipc_plus_3_avg, 0))

# Poverty 
df_pov <- read_excel(paste0(tcd_dir, 'tcd-subnational-results-mpi-2020.xlsx'), sheet = 1) %>% # first sheet MPI by region
  select(c(7:13)) %>% # select columns of mpi by region
  slice(9:n()) %>% # remove header rows
  slice(1:21) # remove NA rows at bottom

colnames(df_pov) <- c('adm1_name', 'mpi_adm0', 'mpi_adm1', 'hcr', 'dep_in', 'vuln_pov', 'sev_pov')

x <- df_pov[df_pov$adm1_name == "Borkou/Tibesti",] # create separate rows for these two regions
x$adm1_name <- 'Borkou'

y <- df_pov[df_pov$adm1_name == "Ennedi Est & Ennedi Ouest",] # create separate rows for these two regions
y$adm1_name <- 'Ennedi Ouest'

df_pov <- rbind(df_pov, x, y)

df_pov$adm1_name[df_pov$adm1_name == "Barh El Gazal"] <- 'Barh-El-Gazel'
df_pov$adm1_name[df_pov$adm1_name == "Borkou/Tibesti"] <- 'Tibesti'
df_pov$adm1_name[df_pov$adm1_name == "Chari Baguirmi"] <- 'Chari-Baguirmi'
df_pov$adm1_name[df_pov$adm1_name == "Ennedi Est & Ennedi Ouest"] <- 'Ennedi Est'
df_pov$adm1_name[df_pov$adm1_name == "Mayo Kebbi Est"] <- 'Mayo-Kebbi Est'
df_pov$adm1_name[df_pov$adm1_name == "Mayo Kebbi Ouest"] <- 'Mayo-Kebbi Ouest'
df_pov$adm1_name[df_pov$adm1_name == "Moyen Chari"] <- 'Moyen-Chari'
df_pov$adm1_name[df_pov$adm1_name == "N’Djaména"] <- "N'Djamena"

df_pov <- df_pov %>% 
  mutate(across(c('mpi_adm0':'sev_pov'), as.numeric))

# Operational presence 
df_op <- read_excel(paste0(tcd_dir, '3w_tcd_june2020.xlsx'), sheet = 1) %>%
  slice(2:n()) # remove hxl tags row

df_op_sum <- df_op %>% # Get the number of activities per adm1
  group_by(Pcode1, Region) %>%
  summarise(num_op = n())%>%
  mutate(num_op = as.numeric(num_op))
