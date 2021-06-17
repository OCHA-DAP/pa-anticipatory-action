library(dplyr)
library(sf)
library(ggplot2)
library(tidyr)
library(tibble)
library(lubridate)
library(zoo)
library(raster)
library(flextable)
library(ggcorrplot)

# -------------------------------------------------------------------------
# Exploring the relation between monthly rainfall and flooding in South Sudan
# -------------------------------------------------------------------------


# Setup -------------------------------------------------------------------

data_dir <- Sys.getenv("AA_DATA_DIR")
data_private_dir <- Sys.getenv("AA_DATA_PRIVATE_DIR")
floodscan_dir <- paste0(data_dir, "/private/processed/ssd/floodscan")
precip_dir = paste0(data_dir, '/public/processed/ssd/chirps/gee_output/')
plot_dir = paste0(data_dir, "/public/exploration/ssd/plots")
floodscan_file_path=paste0(floodscan_dir,"/ssd_floodscan_stats_adm1.csv")

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

########################################################################################
############################# Load and plot monthly precipitation data ######################################
########################################################################################
precip_files <- list.files(path = precip_dir, pattern='SSD_adm1_ucsb-chg-chirps-daily_mean')

df_list_precip <- list()

#load historical observed precipitation data
for (i in 1:length(precip_files)){

  df = read.csv(paste0(precip_dir, precip_files[i]), header=FALSE)
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

#average precipitation across years for the given month
df_precip_avg <- df_precip_all %>%
  group_by(month, Region) %>%
  summarise(month_avg_precip = mean(sum_precip))

df_precip_all <- df_precip_all %>%
  left_join(df_precip_avg, by=c('Region', 'month'))
df_precip_all$anomaly <- df_precip_all$sum_precip - df_precip_all$month_avg_precip

#plot historical monthly precipitation, also showing the average
plt_precip <- ggplot(df_precip_all)+
  geom_bar(aes(x=date, y=sum_precip),fill='lightblue', stat='identity')+
  geom_line(aes(x=date, y=month_avg_precip), color='darkred', size=0.25)+
  facet_grid(rows=vars(Region))+
  theme_bw()+
  labs(x='Date', y='Precipitation (mm)')+
  theme(legend.position = 'bottom',
        axis.text.y = element_text(size=10),)

# # ggsave(paste0(plot_dir, '/chirps_monthly/ssd_monthly_precipitation_mean_1998_2020_adm1.png'), width = 20, height = 15, units = "in", dpi = 300)
#
# #convert month to numeric in order to filter on rainy season
df_precip_all$month_num <- as.numeric(df_precip_all$month)
df_precip_all$pos <- ifelse(df_precip_all$month_num %in% 7:10,ifelse(df_precip_all$anomaly >= 0, "positive anomaly", "negative anomaly"),"outside rainy season")
df_precip_all$pos <- factor(df_precip_all$pos, levels = c("positive anomaly","negative anomaly","outside rainy season"))

#plot anomalies
plt_precip_anom <- ggplot(df_precip_all)+
  geom_bar(aes(x=date, y=anomaly,fill=pos),stat='identity')+
  scale_fill_manual(values=c("#18998F","#C25048", "#cccccc"))+
  scale_x_date(date_breaks = "1 years",date_labels = "%Y")+
  facet_grid(rows=vars(Region))+
  theme_bw()+
  labs(x='Date', y='Precipitation anomaly (mm)')+
  theme(legend.position = 'bottom',
        axis.text.y = element_text(size=10),
        axis.text.x = element_text(angle=90),
        strip.text.y = element_text(size = 5))
# # ggsave(paste0(plot_dir, '/chirps_monthly/ssd_monthly_precipitation_mean_1998_2020_adm1_anomaly.png'), width = 20, height = 15, units = "in", dpi = 300)

#plot only one admin for graph for presentation
plt_precip_anom_jonglei <- ggplot(df_precip_all %>% filter(Region=="Jonglei"))+
  geom_bar(aes(x=date, y=anomaly,fill=pos),stat='identity')+
  scale_fill_manual(values=c("#18998F","#C25048", "#cccccc"))+
  scale_x_date(date_breaks = "1 years",date_labels = "%Y")+
  facet_grid(cols=vars(Region))+
  theme_bw()+
  theme_hm+
  labs(x='Date', y='Precipitation anomaly (mm)')+
  theme(legend.position = 'bottom',
        axis.text.y = element_text(size=10),
        axis.text.x = element_text(angle=90))
# ggsave(paste0(plot_dir, '/chirps_monthly/ssd_monthly_precipitation_mean_1998_2020_jonglei_anomaly.png'), plot=plt_precip_anom_jonglei, width = 14, height = 10, units = "in", dpi = 300)

########################################################################################
############################# Load and plot monthly floodscan data ######################################
########################################################################################
#TODO: now using the mean_cell value. However an e.g. 5 percentile value might be more suitable, but couldn't get this to work in the preprocessing
#TODO: compute rolling average floodscan to remove noise
#TODO: the dry mask hasn't been used for computing these values... should probably add that (but should be part of the .py script)
#TODO: determine flooding events, code from mwi might be reusable: https://github.com/OCHA-DAP/pa-anticipatory-action/blob/1b3bda2e3b3f29a4afa7ab2d121550b469dc7b3a/analyses/malawi/flooding_trigger/mwi-floodscan.md
df_fs = read.csv(floodscan_file_path)

df_fs <- df_fs %>%
  mutate(date = as.Date(date)) %>%
  mutate(month_day = format(date, "%m-%d"))%>%
  mutate(month_year = format(date, "%Y-%m"))%>%
  mutate(date_no_year = as.Date(paste0('1800-',month_day), "%Y-%m-%d")) %>%
  mutate(year = format(date, "%Y"), month = format(date, "%m"))
df_fs$rainy_season <- ifelse(as.numeric(df_fs$month) %in% 7:10,"rainy","not rainy")

#group to month and admin1
df_fs_month <- df_fs %>%
  group_by(month,year, ADM1_EN) %>%
  summarise(max_cell_month = max(max_cell),mean_cell_month = max(mean_cell),percentile_10_month=max(percentile_10)) %>%
  mutate(date = lubridate::as_date(paste0(year, month, '01'), format='%Y%m%d'))
df_fs_month$rainy_season <- ifelse(as.numeric(df_fs_month$month) %in% 7:10,"yes","no")
#sort values by date
df_fs_month <- df_fs_month[order(df_fs_month$year,df_fs_month$month),]

#plot historical values, colored by rainy season
df_fs_bar <- ggplot(df_fs_month)+
  geom_bar(aes(x=date, y=mean_cell_month,fill=rainy_season),stat='identity')+
  scale_fill_manual(values=c(yes="#007CE0", no="#cccccc"))+
  scale_x_date(date_breaks = "3 years",date_labels = "%Y")+
  # facet_wrap(~ADM1_EN,ncol = 3)+
  facet_grid(rows=vars(ADM1_EN))+
  theme_bw()+
  labs(x='Date', y='Mean flooded fraction (%)')+
  theme(legend.position = 'bottom',
        axis.text.y = element_text(size=10),
        axis.text.x = element_text(angle=90),
        strip.text.y = element_text(size = 5))

#plot historical values based on max value within adm1, colored by rainy season
df_fs_bar_max <- ggplot(df_fs_month)+
  geom_bar(aes(x=date, y=max_cell_month,fill=rainy_season),stat='identity')+
  scale_fill_manual(values=c(yes="#007CE0", no="#cccccc"))+
  scale_x_date(date_breaks = "1 years",date_labels = "%Y")+
  facet_wrap(~ADM1_EN)+
  theme_bw()+
  labs(x='Date', y='Precipitation anomaly (mm)',title="Max flooded fraction per admin1")+
  theme(legend.position = 'bottom',
        axis.text.y = element_text(size=10),
        axis.text.x = element_text(angle=90))

#only plot Jonglei region, for graph in presentation
plt_fs_bar_jonglei <- ggplot(df_fs_month %>% filter(ADM1_EN=="Jonglei"))+
  geom_bar(aes(x=date, y=mean_cell_month,fill=rainy_season),stat='identity')+
  scale_fill_manual(values=c(yes="#007CE0", no="#cccccc"))+
  scale_x_date(date_breaks = "1 years",date_labels = "%Y")+
  facet_wrap(~ADM1_EN)+
  theme_bw()+
  labs(x='Date', y='Precipitation anomaly (mm)')+
  theme(legend.position = 'bottom',
        axis.text.y = element_text(size=10),
        axis.text.x = element_text(angle=90))
# ggsave(paste0(plot_dir, '/floodscan/ssd_floodscan_mean_1998_2020_jonglei.png'), plot=plt_fs_bar_jonglei, width = 10, height = 8, units = "in", dpi = 300)

########################################################################################
############################# Correlation precipitation and floodscan data ######################################
########################################################################################
#TODO: analyze correlation per region
#TODO: analyze correlation for extreme values, i.e. flooding events 
#combine the two datasets
df_comb <- df_precip_all %>%
  full_join(df_fs_month, by=c('month', 'year', 'Region'='ADM1_EN'))
df_comb$rainy_season <- ifelse(as.numeric(df_comb$month) %in% 7:10,"yes","no")

df_comb_rainy <- filter(df_comb, rainy_season=="yes")

#scatter plot
plt_scatt_fs_precip <- ggplot(df_comb_rainy, aes(x=df_comb_rainy$sum_precip,y=df_comb_rainy$mean_cell_month))+
  geom_point(alpha=0.5, size=2)+
  facet_wrap(~Region)+
  theme_bw()+
  labs(y='monthly mean fraction of cell flooded', x='monthly precipitation (mm)',title="Correlation precipitation and flooding during the rainy season")


select.me <- c('sum_precip','anomaly','max_cell_month','mean_cell_month')
cor_all=cor(df_comb_rainy[,select.me],df_comb_rainy[,select.me],use='p')

#correlation (across all regions)
plt_cor_all <- ggcorrplot(cor_all,
                          type = "lower",
                          insig = "blank",
                          lab=TRUE,
                          ggtheme=theme_bw())

group_cor <- function(grp, df){
  df_sel <- df %>%
    filter(Region==grp)
  return(cor(df_sel[,select.me], use='p'))
}

#correlation Jonglei
cor_jonglei <- group_cor('Jonglei', df_comb_rainy)
plt_cor_jonglei <- ggcorrplot(cor_jonglei,
                          type = "lower",
                          insig = "blank",
                          lab=TRUE,
                          ggtheme=theme_bw())