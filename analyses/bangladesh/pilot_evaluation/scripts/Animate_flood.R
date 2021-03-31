library(tmap)
library(sf)
library(dplyr)
library(gifski)

#setwd("C:/Users/Hannah/Desktop/pa-anticipatory-action/analyses/bangladesh")

shp_adm <- st_read('data/ADM_Shp/bgd_admbnda_adm4_bbs_20201113.shp')
shp_river <- st_read('data/ADM_Shp/river_extent.shp')
df_fi <- read.csv('data/FE_Results_06_02/ADM4_flood_extent_interpolated.csv')
df_fs <- read.csv('data/FE_Results_06_02/ADM4_flood_extent_sentinel.csv')

shp_fs <- shp_adm %>%
  select(ADM4_PCODE, geometry)%>%
  right_join(df_fs, by='ADM4_PCODE')
  

map <- tm_shape(shp_fs) + tm_fill(col='flooded_fraction', palette = 'GnBu', title='Flooded fraction')+
  tm_facets(along='date')+
  tm_scale_bar()+
  tm_shape(shp_river) + tm_fill(col='#67a2c2')+
  tm_layout(frame=FALSE,
            panel.label.bg.color = '#ffffff',
            panel.label.color = 'black')

tmap_animation(map, 'flood_time_series.gif', delay=75)


