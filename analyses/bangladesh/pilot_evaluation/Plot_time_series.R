library(sf)
library(ggplot2)
library(tmap)
library(dplyr)


# 1. Read in and process the time series and summary data -----------------


setwd("C:/Users/Hannah/Desktop/pa-anticipatory-action/analyses/bangladesh")

df_flooding <- read.csv('data/FE_Results/MAUZ_flood_summary_QA.csv') 
df_flooding <- df_flooding %>%
  mutate(PEAK_SAT = as.Date(df_flooding$PEAK_SAT, format = '%Y-%m-%d')) %>%
  mutate(PEAK_G = as.Date(df_flooding$PEAK_G, format = '%Y-%m-%d')) %>%
  mutate(PEAK_SAT_DAYS = as.integer(PEAK_SAT -
                                      as.Date('2020-06-01', format = '%Y-%m-%d')))%>%
  mutate(PEAK_G_DAYS = as.integer(PEAK_G -
                                    as.Date('2020-06-01', format = '%Y-%m-%d')))%>%
  mutate(MAX_DIFF = abs(MAX_G - MAX_SAT))%>%
  mutate(FWHM_NO = ifelse(FWHM < 200, FWHM, NA))


df_ts_intp <- read.csv('data/FE_Results/MAUZ_flood_extent_interpolated.csv')
df_ts_sent <- read.csv('data/FE_Results/MAUZ_flood_extent_sentinel.csv')

# 2. Identify the edge case mauzas for Gaussian fit --------------------------------

# jUST ADJUST THE FILTER PARAMETER TO GET DIFF SUBSETS OF THE DATA
sel <- df_flooding %>%
  filter(MAX_DIFF > 0.5)%>%
  select(PCODE)

ts_intp <- df_ts_intp %>%
  filter(PCODE %in% sel$PCODE) %>%
  mutate(DATE = as.Date(DATE, format = '%Y-%m-%d'))%>%
  mutate(type = 'Gaussian') %>%
  rename(date = DATE, MAUZ_PCODE = PCODE, flooded_fraction = FLOOD_EXTENT)

ts_sent <- df_ts_sent %>%
  filter(MAUZ_PCODE %in% sel$PCODE)%>%
  mutate(date = as.Date(date, format = '%Y-%m-%d'))%>%
  mutate(type = 'Satellite')

ts = rbind(ts_intp, ts_sent)

ggplot(data=ts, aes(date, flooded_fraction, color=type))+
  geom_line()+
  facet_wrap(~MAUZ_PCODE)+
  theme_minimal()+
  theme(strip.text.x = element_blank())

ggsave('results/max_diff_over_30.png')

