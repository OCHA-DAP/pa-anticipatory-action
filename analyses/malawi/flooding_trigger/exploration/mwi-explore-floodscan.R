library(dplyr)
library(sf)
library(lubridate)
library(readxl)
library(yaml)
library(tmap)
library(ggplot2)
library(stringr)

RAINY_START <- 10
RAINY_END <- 4

# Get overall config params and read in data
data_dir <- Sys.getenv("AA_DATA_DIR")
data_private_dir <- paste0(data_dir, '/private')
config <- read_yaml('../../src/malawi/config.yml')

df_floodscan <- read.csv(paste0(data_private_dir, '/processed/mwi/floodscan/mwi_floodscan_stats_adm2.csv'))

df_floodscan_sel <- df_floodscan %>%
  filter(ADM2_EN %in% c('Chikwawa')) %>%
  select('ADM2_EN', 'mean_cell', 'date') %>%
  mutate(date = as.Date(date)) %>%
  mutate(month_day = format(date, "%m-%d"))%>%
  mutate(date_no_year = as.Date(paste0('1800-',month_day), "%Y-%m-%d"))%>%
  mutate(year = lubridate::year(date)) %>%
  mutate(month = lubridate::month(date)) %>%
  mutate(season_approx = ifelse(month >= RAINY_START, year, 
                                ifelse(month <= RAINY_END, year - 1, 'outside rainy season'))) %>%
  filter(!str_detect(season_approx, 'outside rainy season')) %>%
  group_by(season_approx) %>% 
  mutate(id = row_number())

mean_flood_frac <- mean(df_floodscan_sel$mean_cell)


ggplot(df_floodscan_sel, aes(x=id, y=mean_cell, color=season_approx))+
  geom_line(alpha=0.3)+
  theme_minimal()+
  scale_color_manual(values=rep(c('black'),times=24))+
  geom_hline(yintercept=mean_flood_frac, color = "red", size=1)+
  labs(x='Days into rainy season', y='Mean flooded fraction')+
  theme(legend.position = "none") 

ggsave(paste0(data_dir, '/processed/mwi/plots/flooding/floodscan_overview_nsanje.png'))
