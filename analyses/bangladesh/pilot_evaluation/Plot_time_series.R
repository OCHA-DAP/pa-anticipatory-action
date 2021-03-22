library(sf)
library(ggplot2)
library(tmap)
library(dplyr)


# 1. Read in and process the time series and summary data -----------------


setwd("C:/Users/Hannah/Desktop/pa-anticipatory-action/analyses/bangladesh")

df_flooding <- read.csv('data/FE_Results/June_Aug/MAUZ_flood_summary_QA_survey.csv') 
df_flooding <- df_flooding %>%
  mutate(PEAK_SAT = as.Date(df_flooding$PEAK_SAT, format = '%Y-%m-%d')) %>%
  mutate(PEAK_G = as.Date(df_flooding$PEAK_G, format = '%Y-%m-%d')) %>%
  mutate(PEAK_SAT_DAYS = as.integer(PEAK_SAT -
                                      as.Date('2020-06-01', format = '%Y-%m-%d')))%>%
  mutate(PEAK_G_DAYS = as.integer(PEAK_G -
                                    as.Date('2020-06-01', format = '%Y-%m-%d')))%>%
  mutate(MAX_DIFF = abs(MAX_G - MAX_SAT))%>%
  mutate(FWHM_NO = ifelse(FWHM < 200, FWHM, NA))%>%
  mutate(DIFF_SAT = abs(DIFF_SAT))

shp <- st_read('data/ADM_Shp/selected_distict_mauza.shp') %>%
  select(OBJECTID, geometry) %>%
  filter(OBJECTID %in% df_flooding$PCODE)

df_ts_intp <- read.csv('data/FE_Results/June_Aug/MAUZ_flood_extent_interpolated.csv')
df_ts_sent <- read.csv('data/FE_Results/June_Aug/MAUZ_flood_extent_sentinel.csv')

# 2. Identify the edge case mauzas for Gaussian fit --------------------------------

gaussian_qa <- function(cnd,fname){
  
  sel <- df_flooding %>%
    filter(cnd)%>%
    select(OBJECTID)
  
  # For randomly selecting mauzas
  #sel <- df_flooding[sample(nrow(df_flooding), 56),] %>%
  #  select(PCODE)
  
  ts_intp <- df_ts_intp %>%
    filter(PCODE %in% sel$OBJECTID) %>%
    mutate(DATE = as.Date(DATE, format = '%Y-%m-%d'))%>%
    mutate(type = 'Gaussian') %>%
    rename(date = DATE, MAUZ_PCODE = PCODE, flooded_fraction = FLOOD_EXTENT)
  
  ts_sent <- df_ts_sent %>%
    filter(MAUZ_PCODE %in% sel$OBJECTID)%>%
    mutate(date = as.Date(date, format = '%Y-%m-%d'))%>%
    mutate(type = 'Satellite')
  
  ts = rbind(ts_intp, ts_sent)
  
  ggplot(data=ts, aes(date, flooded_fraction, color=type))+
    geom_line()+
    facet_wrap(~MAUZ_PCODE)+
    theme_classic()+
    theme(strip.text.x = element_blank())+
    labs(y='Flooded fraction', x='Date')
  
  ggsave(paste0('results/fig', fname, '.png'))
  
  shp_sel <- shp %>%
    mutate(HLT = case_when(OBJECTID %in% sel$PCODE ~ 1, TRUE ~ 0))
  
  m <- tm_shape(shp_sel) +
    tm_layout(frame =FALSE) +
    tm_fill(col='HLT', palette='OrRd', legend.show = FALSE)
  
  tmap_save(m, paste0('results/map', fname, '.png'))
}

#gaussian_qa(df_flooding$COV > 5, 'survey_cov_over_5')
gaussian_qa(df_flooding$FWHM > 200, 'survey_fwhm_over_100')
#gaussian_qa(df_flooding$DIFF_SAT > 20, 'diff_peak_over_20')
#gaussian_qa(df_flooding$MAX_DIFF > 0.5, 'max_diff_over_05')

# 3. Explore trends in the uncertainty of fit -----------------------------

ggplot(df_flooding, aes(x=COV, y=MAX_G))+
  geom_point(alpha=0.3)+
  xlim(0, 25)+
  ylim(0,1)+
  geom_smooth(method=lm)+
  theme_classic()+
  labs(y = 'Max Gaussian flooded fraction', x = 'Peak date uncertainty (days)')

