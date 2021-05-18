library(dplyr)
library(sf)
library(ggplot2)
library(tidyr)
library(tibble)
library(lubridate)
library(zoo)
library(raster)
library(flextable)

# -------------------------------------------------------------------------
# Exploring monthly rainfall in South Sudan
# -------------------------------------------------------------------------


# Setup -------------------------------------------------------------------

data_dir <- Sys.getenv("AA_DATA_DIR")
data_private_dir <- Sys.getenv("AA_DATA_PRIVATE_DIR")
floodscan_dir <- paste0(data_dir, "/private/processed/ssd/floodscan")
temp_dir = paste0(data_dir, '/public/processed/ssd/chirps/gee_output/')
plot_dir = paste0(data_dir, "/public/exploration/ssd/plots")
# Monthly precipitation ---------------------------------------------------

precip_files <- list.files(path = temp_dir, pattern='SSD_adm1_ucsb-chg-chirps-daily_mean')

# precip_files <- list.files(path = temp_dir, pattern='mwi_adm1_ucsb-chg-chirps-daily')
df_list_precip <- list()

#settings for plots
theme_hm <- theme_minimal()+
  theme(
    axis.text.y=element_blank(),
    axis.ticks.y=element_blank(),
    legend.position = 'bottom',
    strip.text = element_text(size=16,angle=0),
    axis.text.x = element_text(size=20,angle=90),
    legend.text = element_text(size=24),
    legend.title = element_text(size=24),
    axis.title.x = element_text(size=20),
    axis.title.y = element_text(size=20),
    plot.title = element_text(size=32)
  )

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
# flextable(df_precip_all)
df_precip_all$anomaly <- df_precip_all$sum_precip - df_precip_all$month_avg_precip

# plt_precip <- ggplot(df_precip_all)+
#   geom_bar(aes(x=date, y=sum_precip),fill='lightblue', stat='identity')+
#   geom_line(aes(x=date, y=month_avg_precip), color='darkred', size=0.25)+
#   # facet_wrap(month) +
#   #scale_color_discrete(labels = c("Total monthly precip", "Average total monthly precip"))+
#   facet_grid(rows=vars(Region))+
#   theme_bw()+
#   labs(x='Date', y='Precipitation (mm)')+
#   theme(legend.position = 'bottom',
#         axis.text.y = element_text(size=10),)
# plt_precip
# 
# # ggsave(paste0(plot_dir, '/chirps_monthly/ssd_monthly_precipitation_mean_1998_2020_adm1.png'), width = 20, height = 15, units = "in", dpi = 300)
# 
# #convert month to numeric in order to filter on rainy season
df_precip_all$month_num <- as.numeric(df_precip_all$month)
df_precip_all$pos <- ifelse(df_precip_all$month_num %in% 7:10,ifelse(df_precip_all$anomaly >= 0, "positive anomaly", "negative anomaly"),"outside rainy season")
df_precip_all$pos <- factor(df_precip_all$pos, levels = c("positive anomaly","negative anomaly","outside rainy season"))
# plt_precip_anom <- ggplot(df_precip_all)+
#   geom_bar(aes(x=date, y=anomaly,fill=pos),stat='identity')+
#   scale_fill_manual(values=c("#18998F","#C25048", "#cccccc"))+
#   # geom_line(aes(x=date, y=month_avg_precip), color='darkred', size=0.25)+
#   #scale_color_discrete(labels = c("Total monthly precip", "Average total monthly precip"))+
#   scale_x_date(date_breaks = "1 years",date_labels = "%Y")+
#   facet_grid(rows=vars(Region))+
#   theme_bw()+
#   labs(x='Date', y='Precipitation anomaly (mm)')+
#   theme(legend.position = 'bottom',
#         axis.text.y = element_text(size=10),
#         axis.text.x = element_text(angle=90))
# plt_precip_anom

#plot only one admin for presentation graph
plt_precip_anom_jonglei <- ggplot(df_precip_all %>% filter(Region=="Jonglei"))+
  geom_bar(aes(x=date, y=anomaly,fill=pos),stat='identity')+
  scale_fill_manual(values=c("#18998F","#C25048", "#cccccc"))+
  # geom_line(aes(x=date, y=month_avg_precip), color='darkred', size=0.25)+
  #scale_color_discrete(labels = c("Total monthly precip", "Average total monthly precip"))+
  scale_x_date(date_breaks = "1 years",date_labels = "%Y")+
  facet_grid(cols=vars(Region))+
  theme_bw()+
  theme_hm+
  labs(x='Date', y='Precipitation anomaly (mm)')+
  theme(legend.position = 'bottom',
        axis.text.y = element_text(size=10),
        axis.text.x = element_text(angle=90))
plt_precip_anom_jonglei
ggsave(paste0(plot_dir, '/chirps_monthly/ssd_monthly_precipitation_mean_1998_2020_jonglei_anomaly.png'), plot=plt_precip_anom_jonglei, width = 14, height = 10, units = "in", dpi = 300)



# ggsave(paste0(plot_dir, '/chirps_monthly/ssd_monthly_precipitation_mean_1998_2020_adm1_anomaly.png'), width = 20, height = 15, units = "in", dpi = 300)
df_fs = read.csv(paste0(floodscan_dir,"/ssd_floodscan_stats_adm1.csv"))

df_fs <- df_fs %>%
  mutate(date = as.Date(date)) %>%
  mutate(month_day = format(date, "%m-%d"))%>%
  mutate(month_year = format(date, "%Y-%m"))%>%
  mutate(date_no_year = as.Date(paste0('1800-',month_day), "%Y-%m-%d")) %>%
  mutate(year = format(date, "%Y"), month = format(date, "%m"))

df_fs$rainy_season <- ifelse(as.numeric(df_fs$month) %in% 7:10,"rainy","not rainy")

# flextable(head(df_fs,5))

df_fs_month <- df_fs %>% 
  group_by(month,year, ADM1_EN) %>%
  summarise(max_cell_month = max(max_cell),mean_cell_month = max(mean_cell),percentile_10_month=max(percentile_10)) %>% 
  mutate(date = lubridate::as_date(paste0(year, month, '01'), format='%Y%m%d'))
df_fs_month$rainy_season <- ifelse(as.numeric(df_fs_month$month) %in% 7:10,"yes","no")


df_fs_month <- df_fs_month[order(df_fs_month$year,df_fs_month$month),]

# plt_fs <- ggplot(df_fs, aes(x=date, y=mean_cell))+#, group=ADM1_EN)) +
#   geom_line(aes(color=ADM1_EN))+
#   facet_wrap(~ADM1_EN)+
#   labs(y='mean fraction of cell flooded', x='Date')+
#   theme_bw()+
#   theme(legend.position = 'bottom')+
#   scale_x_date(date_labels = "%Y") #+
# plt_fs
# 
plt_fs <- ggplot(df_fs_month, aes(x=date, y=mean_cell_month,color=rainy_season))+
  scale_color_manual(values=c(yes="#C25048", no="#cccccc"))+
  facet_wrap(~ADM1_EN)+
  geom_line(aes(group = 1))+
  labs(y='mean fraction of cell flooded', x='Date')+
  theme_bw()+
  theme(legend.position = 'bottom')+
  scale_x_date(date_labels = "%Y")
plt_fs

#only plot one region, for presentation graph
plt_fs_jonglei <- ggplot(df_fs_month %>% filter(ADM1_EN=="Jonglei"), aes(x=date, y=mean_cell_month,color=rainy_season))+#, group=ADM1_EN)) +
  scale_color_manual(values=c(yes="#007CE0", no="#cccccc"))+
  facet_wrap(~ADM1_EN)+
  geom_line(aes(group = 1))+
  labs(y='mean fraction of cell flooded', x='Date')+
  theme_bw()+
  theme_hm+
  theme(legend.position = 'bottom')+
  scale_x_date(date_breaks = "1 years", date_labels = "%Y")
plt_fs_jonglei

ggsave(paste0(plot_dir, '/floodscan/ssd_floodscan_mean_1998_2020_jonglei.png'), plot=plt_fs_jonglei, width = 10, height = 8, units = "in", dpi = 300)



# plt_fs <- ggplot(df_fs_month, aes(x=date, y=percentile_10_month,color=rainy_season))+#, group=ADM1_EN)) +
#   scale_color_manual(values=c(yes="#C25048", no="#cccccc"))+#(values = c(apple = "red", avocado = "green", ...)
#   facet_wrap(~ADM1_EN)+
#   geom_line(aes(group = 1))+
#   labs(y='percentile fraction of cell flooded', x='Date')+
#   theme_bw()+
#   theme(legend.position = 'bottom')+
#   scale_x_date(date_labels = "%Y") 
# plt_fs

# df_fs_bar <- ggplot(df_fs_month)+
#   geom_bar(aes(x=date, y=mean_cell_month,fill=rainy_season),stat='identity')+
#   scale_fill_manual(values=c(yes="#C25048", no="#cccccc"))+
#   # geom_line(aes(x=date, y=month_avg_precip), color='darkred', size=0.25)+
#   #scale_color_discrete(labels = c("Total monthly precip", "Average total monthly precip"))+
#   scale_x_date(date_breaks = "1 years",date_labels = "%Y")+
#   # facet_grid(rows=vars(ADM1_EN))+
#   facet_wrap(~ADM1_EN)+
#   theme_bw()+
#   labs(x='Date', y='Precipitation anomaly (mm)')+
#   theme(legend.position = 'bottom',
#         axis.text.y = element_text(size=10),
#         axis.text.x = element_text(angle=90))
# df_fs_bar


# plt_fs_month <- ggplot(df_fs_month, aes(x=date, y=max_cell_month))+#, group=ADM1_EN)) +
#   geom_line(aes(color=ADM1_EN))+
#   facet_wrap(~ADM1_EN)+
#   labs(y='max fraction of cell flooded', x='Date')+
#   theme_bw()+
#   theme(legend.position = 'bottom')+
#   scale_x_date(date_labels = "%Y") #+
# plt_fs_month
# 
# 
df_comb <- df_precip_all %>%
  full_join(df_fs_month, by=c('month', 'year', 'Region'='ADM1_EN'))
df_comb$rainy_season <- ifelse(as.numeric(df_comb$month) %in% 7:10,"yes","no")
# 
# plt_fs_precip <- ggplot(df_comb, aes(x=df_comb$sum_precip,y=df_comb$max_cell_month))+
#   geom_point(alpha=0.5, size=2)+
#   facet_wrap(~Region)+
#   theme_bw()+
#   labs(y='monthly max fraction of cell flooded', x='monthly precipitation (mm)')
# 
# plt_fs_precip
# 
# plt_fs_precip <- ggplot(df_comb, aes(x=df_comb$sum_precip,y=df_comb$mean_cell_month))+
#   geom_point(alpha=0.5, size=2)+
#   facet_wrap(~Region)+
#   theme_bw()+
#   labs(y='monthly mean fraction of cell flooded', x='monthly precipitation (mm)')
# 
# plt_fs_precip
# 
df_comb_rainy <- filter(df_comb, rainy_season=="yes")
plt_fs_precip <- ggplot(df_comb_rainy, aes(x=df_comb_rainy$sum_precip,y=df_comb_rainy$mean_cell_month))+
  geom_point(alpha=0.5, size=2)+
  facet_wrap(~Region)+
  theme_bw()+
  labs(y='monthly mean fraction of cell flooded', x='monthly precipitation (mm)')+
  ggtitle("rainy season")

plt_fs_precip

plt_fs_precip <- ggplot(df_comb_rainy, aes(x=df_comb_rainy$mean_cell_month,y=df_comb_rainy$sum_precip))+
  geom_point(alpha=0.5, size=2)+
  facet_wrap(~Region)+
  theme_bw()+
  labs(y='monthly mean fraction of cell flooded', x='monthly precipitation (mm)')+
  ggtitle("rainy season")

plt_fs_precip
# 
# print(cor(df_comb$sum_precip,df_comb$mean_cell_month))

# plt_cor_all <- ggcorrplot(cor_central, 
#                           type = "lower", 
#                           insig = "blank", 
#                           lab=TRUE,
#                           ggtheme=theme_bw())

# plot_scatter(df_comb$sum_precip, df_comb$max_cell_month, 'Number of dry spell days', 'Mean ASI')

  # annotate("rect", xmin = as.Date('1800-07-01'), xmax = as.Date('1800-10-01'), ymin = 0, ymax = 100, 
  #          alpha = .25)
# df_fs = read.csv(paste0(floodscan_dir,"/ssd_floodscan_stats_adm1.csv"))
# # flextable(df_fs)
# plt_fs <- ggplot(df_fs)+
#   geom_line(aes(x=date, y=max_cell))+
#   # geom_line(aes(x=date, y=month_avg_precip), color='darkred', size=0.25)+
#   # facet_wrap(month) +
#   #scale_color_discrete(labels = c("Total monthly precip", "Average total monthly precip"))+
#   facet_grid(rows=vars(ADM1_EN))+
#   theme_bw()+
#   labs(x='Date', y='Flooded fraction (mm)')+
#   theme(legend.position = 'bottom',
#         axis.text.y = element_text(size=10),)
# plt_fs

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


