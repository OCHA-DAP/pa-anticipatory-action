library(sf)
library(ggplot2)
library(tmap)
library(dplyr)

setwd("C:/Users/Hannah/Desktop/pa-anticipatory-action/analyses/bangladesh")


# Histograms --------------------------------------------------------------


df_flooding <- read.csv('data/FE_Results_06_02/MAUZ_flood_summary_QA.csv') %>%
  mutate(PEAK_SAT = as.Date(df_flooding$PEAK_SAT, format = '%Y-%m-%d')) %>%
  mutate(PEAK_G = as.Date(df_flooding$PEAK_G, format = '%Y-%m-%d')) %>%
  mutate(PEAK_SAT_DAYS = as.integer(PEAK_SAT -
           as.Date('2020-06-01', format = '%Y-%m-%d')))%>%
  mutate(PEAK_G_DAYS = as.integer(PEAK_G -
           as.Date('2020-06-01', format = '%Y-%m-%d')))%>%
  mutate(MAX_DIFF = MAX_G - MAX_SAT)%>%
  mutate(FWHM_NO = ifelse(FWHM < 100, FWHM, NA))

ggplot(df_flooding, aes(x=MAX_G)) +
  geom_histogram(fill="#69b3a2", color="#e9ecef", alpha=0.9)+
  #geom_bar(stat='count', fill="#69b3a2", color="#e9ecef", alpha=0.9)+
  theme_bw()+
  ylab('Count')+
  xlab('Maximum Gaussian flooded fraction')+
  theme(text = element_text(size = 20))
  #scale_x_continuous(trans = 'log10')+
  #labs(caption='Log x scale')

ggsave('results/figures/gaussian-summary/max-gaus.png')


# Choropleth maps ---------------------------------------------------------

shp_flooding <- st_read('data/ADM_Shp/selected_distict_mauza.shp') %>%
  select(OBJECTID, geometry)%>%
  right_join(df_flooding, by=c('OBJECTID'='PCODE'))

m <- tm_shape(shp_flooding) +
  tm_layout(frame = FALSE) +
  tm_fill(col='FWHM_NO', 
          palette='BuPu',
          #style='jenks', 
          title='FWHM\n(Remove above 100)')

tmap_save(m, 'results/figures/gaussian-summary/fwhmno100-map.png')
