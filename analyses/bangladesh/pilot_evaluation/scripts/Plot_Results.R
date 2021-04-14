library(sf)
library(ggplot2)
library(tmap)
library(dplyr)

data_dir <- Sys.getenv("AA_DATA_DIR")
bgd_dir <- paste0(data_dir, '/exploration/bangladesh')

# Histograms --------------------------------------------------------------

df_flooding <- read.csv(paste0(bgd_dir, '/FE_Results/June_Aug/MAUZ_flood_summary_QA.csv')) 
df_flooding <- df_flooding %>%
  mutate(PEAK_SAT = as.Date(df_flooding$PEAK_SAT, format = '%Y-%m-%d')) %>%
  mutate(PEAK_G = as.Date(df_flooding$PEAK_G, format = '%Y-%m-%d')) %>%
  mutate(PEAK_SAT_DAYS = as.integer(PEAK_SAT -
           as.Date('2020-06-01', format = '%Y-%m-%d')))%>%
  mutate(PEAK_G_DAYS = as.integer(PEAK_G -
           as.Date('2020-06-01', format = '%Y-%m-%d')))%>%
  mutate(MAX_DIFF = MAX_G - MAX_SAT)%>%
  mutate(FWHM = ifelse(FWHM < 200, FWHM, NA))%>%
  mutate(COV = ifelse(COV < 50, FWHM, NA))

ggplot(df_flooding, aes(x=RMSE)) +
  geom_histogram(fill="#69b3a2", color="#e9ecef", alpha=0.9)+
  theme_bw()+
  ylab('Count')+
  xlab('RMSE')+
  theme(text = element_text(size = 20))

# ggsave('results/rmse.png')


# Choropleth maps ---------------------------------------------------------

shp_flooding <- st_read(paste0(bgd_dir, '/ADM_Shp/selected_distict_mauza.shp')) %>%
  select(OBJECTID, geometry)%>%
  right_join(df_flooding, by=c('OBJECTID'='PCODE'))

m <- tm_shape(shp_flooding) +
  tm_layout(frame =FALSE) +
  tm_fill(col='PEAK_G',
          colorNA = '#eb4034',
          palette='BuPu',
          style='jenks', 
          title='Satellite-derived\npeak flooding date')

# tmap_save(m, 'results/survey_max_sat.png')
