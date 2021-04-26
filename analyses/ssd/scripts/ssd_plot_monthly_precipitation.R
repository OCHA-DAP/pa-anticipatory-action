library(dplyr)
library(sf)
library(ggplot2)
library(tidyr)
library(tibble)
library(lubridate)
library(zoo)
library(raster)

# -------------------------------------------------------------------------
# Exploring monthly rainfall in South Sudan
# -------------------------------------------------------------------------


# Setup -------------------------------------------------------------------

data_dir <- Sys.getenv("AA_DATA_DIR")
temp_dir = paste0(data_dir, '/processed/ssd/chirps/gee_output/')
plot_dir = paste0(data_dir, "/processed/ssd/plots")
# Monthly precipitation ---------------------------------------------------

precip_files <- list.files(path = temp_dir, pattern='SSD_adm1_ucsb-chg-chirps-daily_mean')
# precip_files <- list.files(path = temp_dir, pattern='mwi_adm1_ucsb-chg-chirps-daily')
df_list_precip <- list()

for (i in 1:length(precip_files)){
  
  df = read.csv(paste0(temp_dir, precip_files[i]), header=FALSE)
  df = as.data.frame(t(df))
  
  n<-dim(df)[1]
  #TODO: find a way to generalize this.. the number of columns seems to depend on the country (dont understand why)
  names <- df[(n-5),]
  colnames(df) <- names
  #TODO: find a way to generalize this..
  df <- df[1:(n-13),]
  df_precip <- df %>%
    rename(date = ADM1_EN)%>%
    mutate(date = substr(date, 1,8))
  
  df_list_precip[[i]] <- df_precip
  
}

df_precip_all = do.call(rbind, df_list_precip)

df_precip_all <- df_precip_all %>%
  gather('Region', 'Precip', -date)%>%
  mutate(Precip = as.numeric(Precip))%>%
  mutate(date = lubridate::as_date(date, format='%Y%m%d'))%>%
  mutate(month = format(date, "%m"), year = format(date, "%Y")) %>%
  group_by(month, year, Region) %>%
  summarise(sum_precip = sum(Precip)) %>%
  mutate(date = lubridate::as_date(paste0(year, month, '01'), format='%Y%m%d'))

df_precip_avg <- df_precip_all %>%
  group_by(month, Region) %>%
  summarise(month_avg_precip = mean(sum_precip))

df_precip_all <- df_precip_all %>%
  left_join(df_precip_avg, by=c('Region', 'month'))

df_precip_all$anomaly <- df_precip_all$sum_precip - df_precip_all$month_avg_precip

plt_precip <- ggplot(df_precip_all)+
  geom_bar(aes(x=date, y=sum_precip),fill='lightblue', stat='identity')+
  geom_line(aes(x=date, y=month_avg_precip), color='darkred', size=0.25)+
  # facet_wrap(month) +
  #scale_color_discrete(labels = c("Total monthly precip", "Average total monthly precip"))+
  facet_grid(rows=vars(Region))+
  theme_bw()+
  labs(x='Date', y='Precipitation (mm)')+
  theme(legend.position = 'bottom',
        axis.text.y = element_text(size=10),)
plt_precip

# ggsave(paste0(plot_dir, '/chirps_monthly/ssd_monthly_precipitation_mean_1998_2020_adm1.png'), width = 20, height = 15, units = "in", dpi = 300)

#convert month to numeric in order to filter on rainy season
df_precip_all$month_num <- as.numeric(df_precip_all$month)
df_precip_all$pos <- ifelse(df_precip_all$month_num %in% 7:10,ifelse(df_precip_all$anomaly >= 0, "positive anomaly", "negative anomaly"),"outside rainy season")
df_precip_all$pos <- factor(df_precip_all$pos, levels = c("positive anomaly","negative anomaly","outside rainy season"))
plt_precip_anom <- ggplot(df_precip_all)+
  geom_bar(aes(x=date, y=anomaly,fill=pos),stat='identity')+
  scale_fill_manual(values=c("#18998F","#C25048", "#cccccc"))+
  # geom_line(aes(x=date, y=month_avg_precip), color='darkred', size=0.25)+
  #scale_color_discrete(labels = c("Total monthly precip", "Average total monthly precip"))+
  scale_x_date(date_breaks = "1 years",date_labels = "%Y")+
  facet_grid(rows=vars(Region))+
  theme_bw()+
  labs(x='Date', y='Precipitation anomaly (mm)')+
  theme(legend.position = 'bottom',
        axis.text.y = element_text(size=10),
        axis.text.x = element_text(angle=90))
plt_precip_anom

ggsave(paste0(plot_dir, '/chirps_monthly/ssd_monthly_precipitation_mean_1998_2020_adm1_anomaly.png'), width = 20, height = 15, units = "in", dpi = 300)

#for one region, to make it the info easier to process
# df_precip_ce <- df_precip_all %>% filter(Region == "Central Equatoria")
# flextable(df_precip_ce)
# plt_precip_anom <- ggplot(df_precip_ce %>% filter(between(month, 7,10)))+
#   geom_bar(aes(x=date, y=anomaly,fill=pos),stat='identity')+
#   # geom_line(aes(x=date, y=month_avg_precip), color='darkred', size=0.25)+
#   #scale_color_discrete(labels = c("Total monthly precip", "Average total monthly precip"))+
#   scale_x_date(date_breaks = "1 years",date_labels = "%Y")+
#   # facet_grid(rows=vars(month))+
#   theme_bw()+
#   labs(x='Date', y='Precipitation anomaly (mm)')+
#   theme(legend.position = 'bottom',
#         axis.text.y = element_text(size=10),
#         axis.text.x = element_text(angle=90))
# plt_precip_anom


