library(dplyr)
library(sf)
library(ggplot2)


# -------------------------------------------------------------------------
# Exploring agricultural stress and related impacts in Malawi
# -------------------------------------------------------------------------


# Setup -------------------------------------------------------------------

data_dir <- Sys.getenv("AA_DATA_DIR")
shapefile_path <- paste0(data_dir, "/raw/malawi/Shapefiles/mwi_adm_nso_20181016_shp")

shp_adm2 <- st_read(paste0(shapefile_path, "/mwi_admbnda_adm2_nso_20181016.shp"))
shp_adm1 <- st_read(paste0(shapefile_path, "/mwi_admbnda_adm1_nso_20181016.shp"))

df_dryspells <- read.csv(paste0(data_dir, '/processed/malawi/dry_spells/dry_spells_during_rainy_season_list_2000_2020_mean_back.csv'))

df_crop <- read.csv(paste0(data_dir, '/exploration/malawi/crop_production/agriculture-and-rural-development_mwi.csv'))
df_asi <- read.csv(paste0(data_dir, '/exploration/malawi/ASI/malawi_asi_dekad.csv'))
df_globalipc <- read.csv(paste0(data_dir, '/processed/malawi/GlobalIPCProcessed/malawi_globalipc_admin2.csv'))
df_fewsnet <- read.csv(paste0(data_dir, '/processed/malawi/FewsNetWorldPop/malawi_fewsnet_worldpop_admin2.csv'))

# Crop production ---------------------------------------------------------

# This data from the World Bank contains information on agriculture and rural development in Malawi.
# Values are reported annually at a country-level. Latest data is 2017. 

# Because of the clear trend in increasing crop production over time, it doesn't look very meaningful
# to identify years with low crop production. We need to know how the crop production deviates from what 
# is expected, and this data isn't really telling us that. 
# Looking at cereal yield per hectare may be more helpful in understanding 
# when an agricultural shock took place?

# How would this line up with the timing of dry spells? 
# Since the data is only annual, we probably want to think about dry spells for the growing season from the 
# previous year. 

filter <- 'Cereal yield (kg per hectare)'

df_crop_sel <- df_crop %>%
  filter(Indicator.Name == filter) %>%
  mutate(Year = as.numeric(Year))%>%
  mutate(Value = as.numeric(Value))%>%
  filter(Year > 2000)
 
# Get the years with lowest 1/3 yield or production
# These are still mostly just the earliest years...
low_years <- df_crop_sel %>% filter(Value < quantile(df_crop_sel$Value, 0.33))

ggplot(df_crop_sel) +
  geom_line(aes(x=Year, y=Value)) +
  labs(y=filter, title='Years with lower 1/3rd cereal yield in Malawi')+
  theme_minimal() +
  geom_vline(xintercept=as.numeric(low_years$Year), colour="red", linetype=2)


# Food insecurity ---------------------------------------------------------

# Identify population in IPC 3+ across regions. 
# Which years show a food security crisis?

# We're looking at the current situation numbers, which are provided 4X/year
# from 2009 to 2021. Thinking about the timing of the growing season, we would perhaps 
# expect to see evidence of poor food insecurity for the July values if there was poor crop production
# potentially resulting from a drought or dry spell. 

df_fewsnet_sel <- df_fewsnet %>%
  mutate(ipc3_plus = CS_3 + CS_4 + CS_5) %>%
  select('ADMIN1', 'ADMIN2', 'date', 'ipc3_plus')%>%
  group_by(ADMIN1, date)%>%
  summarise(tot = sum(ipc3_plus))%>%
  mutate(date= as.Date(date))

ggplot(df_fewsnet_sel, aes(x=date, y=tot, group=ADMIN1))+
  geom_line(aes(color=ADMIN1))+
  theme_minimal()+
  labs(x='Date', y='Population', title='Total population in IPC Phase 3+ by region in Malawi')


# Agricultural stress index -----------------------------------------------
  
# What is 'Area under National Administration'?
# Why do some of the values (which are percentages) go above 1?
# We're talking about ASI, but it seems as though the index is really just 
# the % area with VHI (vegetation health index) less than 35
# What is the VHI based on?

regions <- c('Central Region', 'Northern Region', 'Southern Region')

df_asi_sel <- df_asi %>%
  filter(Province %in% regions) %>%
  mutate(Date = as.Date(Date)) %>%
  filter(Date > '2000-01-01')

ggplot(df_asi_sel, aes(x=Date, y=Data)) +
  geom_line()+
  facet_wrap(~Province)+
  labs(y='% of area with mean VHI < 35', title='Agricultural stress in regions in Malawi')+
  theme_minimal()


# Historical dry spells ---------------------------------------------------

# Aggregate this data to get the total number of dry spells per region
# in each growing season. Pretty much all the dry spells are in the Southern region.

df_dryspells_agg <- df_dryspells %>%
  group_by(region, season_approx) %>%
  summarise(num_ds= n())

# What about the total number of dry spell days in each region per growing season?
df_dryspells_days <- df_dryspells %>%
  group_by(region, season_approx)%>%
  summarise(days_ds=sum(dry_spell_duration))

# What about dry spells that started in the key flowering time?