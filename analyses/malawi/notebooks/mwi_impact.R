library(dplyr)
library(sf)
library(ggplot2)
library(tidyr)
library(tibble)
library(lubridate)
library(zoo)
library(raster)
library(ggcorrplot)

# -------------------------------------------------------------------------
# Exploring agricultural stress and related impacts in Malawi
# -------------------------------------------------------------------------


# Setup -------------------------------------------------------------------

data_dir <- Sys.getenv("AA_DATA_DIR")
shapefile_path <- paste0(data_dir, "/raw/malawi/Shapefiles/mwi_adm_nso_20181016_shp")

shp_adm2 <- st_read(paste0(shapefile_path, "/mwi_admbnda_adm2_nso_20181016.shp"))
shp_adm1 <- st_read(paste0(shapefile_path, "/mwi_admbnda_adm1_nso_20181016.shp"))

df_dryspells <- read.csv(paste0(data_dir, '/processed/malawi/dry_spells/dry_spells_during_rainy_season_list_2000_2020_mean_back.csv'))
df_dryspells_px <- read.csv(paste0(data_dir, '/processed/malawi/dry_spells/ds_counts_per_pixel_adm1.csv'))

df_crop <- read.csv(paste0(data_dir, '/exploration/malawi/crop_production/agriculture-and-rural-development_mwi.csv'))
df_asi <- read.csv(paste0(data_dir, '/exploration/malawi/ASI/malawi_asi_dekad.csv'))
df_globalipc <- read.csv(paste0(data_dir, '/processed/malawi/GlobalIPCProcessed/malawi_globalipc_admin2.csv'))
df_fewsnet <- read.csv(paste0(data_dir, '/processed/malawi/FewsNetWorldPop/malawi_fewsnet_worldpop_admin2.csv'))
df_price <- read.csv(paste0(data_dir, '/exploration/malawi/crop_production/wfp_food_prices_malawi.csv'))

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
  mutate(rollavg = zoo::rollmean(Value, k = 5, fill = NA, align='left'))%>%
  mutate(low = ifelse(Value<rollavg, 1, 0)) %>%
  filter(Year > 1999)

# Get years where production is lower than 5-year rolling avg of prev years 
low_years <- df_crop_sel %>%
  filter(low == 1)

plt_crop <- ggplot(df_crop_sel) +
  geom_bar(stat='identity', aes(x=Year, y=Value), alpha=0.5) +
  geom_line(aes(x=Year, y=rollavg), alpha=0.7)+
  labs(y=filter)+
  theme_minimal() +
  annotate("text", x = as.numeric(low_years$Year), y = low_years$Value+100, color='red',label = as.numeric(low_years$Year), size=2)


# Crop prices -------------------------------------------------------------

# Drop the hxl row in the top
df_price <- df_price[-1,]


# How have the maize prices changed?
sel <- df_price %>%
  filter(cmname == 'Maize - Retail')%>%
  dplyr::select(date, cmname, price, admname)%>%
  mutate(date = as.Date(date))%>%
  mutate(price = as.numeric(price))%>%
  group_by(date, admname)%>%
  summarise(med_price = median(price))

plot(diff(sel$med_price, lag=2))

# As is the prices trend with inflation (I'm assuming), so maybe it'll work to difference
# the series to identify price peaks that would results from dry spells?

Box.test(diff(sel$med_price), lag=1, type="Ljung-Box")

plt_price <- ggplot(sel, aes(x=date, y=med_price, group=admname))+
  geom_line(aes(color=admname))+
  labs(x='Date', y='Median price', color='Region')+
  theme_minimal()

# Food insecurity ---------------------------------------------------------

# Identify population in IPC 3+ across regions. 
# Which years show a food security crisis?

# We're looking at the current situation numbers, which are provided 4X/year
# from 2009 to 2021. Thinking about the timing of the growing season, we would perhaps 
# expect to see evidence of poor food insecurity for the July values if there was poor crop production
# potentially resulting from a drought or dry spell. 

df_fewsnet_sel <- df_fewsnet %>%
  mutate(ipc3_plus = CS_3 + CS_4 + CS_5) %>%
  dplyr::select('ADMIN1', 'ADMIN2', 'date', 'ipc3_plus')%>%
  group_by(ADMIN1, date)%>%
  summarise(tot = sum(ipc3_plus))%>%
  mutate(date= as.Date(date))%>%
  mutate(year = lubridate::year(date))%>%
  mutate(month_day = format(date, "%m-%d"))%>%
  mutate(date_no_year = as.Date(paste0('1800-',month_day), "%Y-%m-%d"))%>%
  mutate(tot = tot/1000000)

high_dates <- df_fewsnet_sel %>%
  group_by(date) %>%
  summarise(tot = sum(tot))%>%
  filter(tot>0)

plt_fewsnet <- ggplot(df_fewsnet_sel, aes(x=date_no_year, y=tot, group=ADMIN1))+
  geom_bar(stat = 'identity', aes(fill=ADMIN1))+
  facet_wrap(~year)+
  scale_x_date(date_labels = "%b")+
  theme_bw()+
  theme(legend.position = 'bottom')+
  labs(x='Date', y='Population IPC 3+ (millions)', fill='Region')#+
  #annotate("text", x = high_dates$date, y = high_dates$tot+500000, label = substring(high_dates$date, 0, 7), size=2, angle=45)

# Get the total IPC 3+ population associated with the PREVIOUS season
# Just shift everything back by 1 year - we'll assume that IPC 3+ pops from Jan - Dec
# are most impacted by the growing season from the previous year
# Remember that pop is in MILLIONS
df_fewsnet_season <- df_fewsnet_sel %>%
  mutate(season_approx = year -1) %>%
  group_by(season_approx, ADMIN1)%>%
  summarise(total_ipc3plus = sum(tot))


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
  filter(Date > '2000-01-01')%>%
  mutate(month_day = format(Date, "%m-%d"))%>%
  mutate(date_no_year = as.Date(paste0('1800-',month_day), "%Y-%m-%d"))

plt_asi <- ggplot(df_asi_sel, aes(x=date_no_year, y=Data, group=Province)) +
  geom_line(aes(color=Province))+
  facet_wrap(~Year)+
  labs(y='% of area with mean VHI < 35', x='Date')+
  theme_bw()+
  theme(legend.position = 'bottom')+
  scale_x_date(date_labels = "%b")+
  annotate("rect", xmin = as.Date('1800-07-01'), xmax = as.Date('1800-10-01'), ymin = 0, ymax = 100, 
           alpha = .25)

# Get the mean ASI by region during each season approx (Oct-June)
df_asi_sel_mean <- df_asi_sel %>%
  mutate(season_approx = ifelse(df_asi_sel$Month >= 10, df_asi_sel$Year, ifelse(df_asi_sel$Month <= 7, df_asi_sel$Year - 1, 'outside rainy season')))%>%
  filter(season_approx!='outside rainy season') %>%
  dplyr::select('Province', 'Data', 'season_approx') %>%
  group_by(Province, season_approx) %>%
  summarise(avg_asi = mean(Data)) %>%
  mutate(season_approx = as.numeric(season_approx))%>%
  filter(season_approx>1999)%>%
  mutate(Province = substring(Province, 0, nchar(Province)-7))

# Get the max ASI by region during each season approx (Oct-June)
df_asi_sel_max <- df_asi_sel %>%
  mutate(season_approx = ifelse(df_asi_sel$Month >= 10, df_asi_sel$Year, ifelse(df_asi_sel$Month <= 7, df_asi_sel$Year - 1, 'outside rainy season')))%>%
  filter(season_approx!='outside rainy season') %>%
  dplyr::select(Province, Data, season_approx) %>%
  group_by(Province, season_approx) %>%
  summarise(max_asi = max(Data)) %>%
  mutate(season_approx = as.numeric(season_approx))%>%
  filter(season_approx>1999)%>%
  mutate(Province = substring(Province, 0, nchar(Province)-7))


# WRSI --------------------------------------------------------------------

# Preprocessed in the mwi_wrsi_process.R file,
# using the outputs from the GeoWRSI software 
wrsi_dir <- paste0(data_dir, '/exploration/malawi/wrsi/')
#wrsi_mean <- read.csv(paste0(wrsi_dir, 'wrsi_mean_adm1.csv'))

wrsi_min <- read.csv(paste0(wrsi_dir, 'wrsi_min_adm1.csv'))
wrsi_min[sapply(wrsi_min, is.infinite)] <- NA

wrsi_plt <- wrsi_min %>%
  ggplot(aes(x=dekad, y=wrsi, group=ID))+
  geom_line(aes(color=ID))+
  facet_wrap(~year)+
  theme_bw()+
  theme(legend.position = 'bottom')+
  labs(x='Dekad', y='WRSI', color='Region')

#wrsi_mean_na <- wrsi_mean %>% drop_na()
wrsi_min_na <- wrsi_min %>% drop_na()

# Get the min WRSI by season by region
# Assuming that the 20th dekad will always be in the dry season 
df_wrsi_season <- wrsi_min_na %>%
  mutate(season_approx = ifelse(wrsi_min_na$dekad < 20, wrsi_min_na$year -1, wrsi_min_na$year))%>%
  group_by(season_approx, ID) %>%
  summarise(min_wrsi = min(wrsi))

# Monthly temperature -----------------------------------------------------

temp_dir = paste0(data_dir, '/processed/malawi/dry_spells/gee_output/')

temp_files <- list.files(path = temp_dir, pattern='mwi_adm1_ecmwf-era5-monthly_median')

df_list <- list()

for (i in 1:length(temp_files)){
  
  df = read.csv(paste0(temp_dir, temp_files[i]), header=FALSE)
  df = as.data.frame(t(df))
  
  n<-dim(df)[1]
  names <- df[(n-8),]
  colnames(df) <- names
  df <- df[1:(n-12),]
  
  df_temp <- df %>%
    rename(date = ADM1_EN)%>%
    mutate(date = substr(date, 1,6))
  
  df_list[[i]] <- df_temp
  
}

df_temp_all = do.call(rbind, df_list)
df_temp_all <- df_temp_all %>%
  gather('Region', 'Temp', -date)%>%
  mutate(date = paste0(date, '01'))%>%
  mutate(date = lubridate::as_date(date, format='%Y%m%d'))%>%
  mutate(month = format(date, "%m"), year = format(date, "%Y")) %>%
  group_by(month, year, Region) %>%
  mutate(Temp = as.numeric(Temp))%>%
  mutate(Temp = Temp - 273.15)%>%
  mutate(month = as.numeric(month), year = as.numeric(year))

df_temp_avg <- df_temp_all %>%
  group_by(month, Region) %>%
  summarise(month_avg_temp = mean(Temp))

df_temp_all <- df_temp_all %>%
  left_join(df_temp_avg, by=c('Region', 'month'))

plt_temp <- ggplot(df_temp_all)+
  geom_bar(aes(x=date, y=Temp), stat='identity',fill='lightpink')+
  geom_line(aes(x=date, y=month_avg_temp), color='darkblue', size=0.25)+
  facet_grid(rows=vars(Region))+
  theme_bw()+
  labs(x='Date', y='Temperature (C)')+
  theme(legend.position = 'bottom')+
  coord_cartesian(ylim=c(15,30))

df_temp_season <- df_temp_all %>%
  as.data.frame()%>%
  mutate(season_approx = ifelse(df_temp_all$month >= 10, df_temp_all$year, ifelse(df_temp_all$month <= 7, df_temp_all$year - 1, 'outside rainy season')))%>%
  filter(season_approx!='outside rainy season') %>%
  group_by(season_approx, Region)%>%
  summarise(avg_season_temp = mean(Temp))%>%
  mutate(season_approx = as.numeric(season_approx))


# Monthly precipitation ---------------------------------------------------

precip_files <- list.files(path = temp_dir, pattern='mwi_adm1_ucsb-chg-chirps-daily')

df_list_precip <- list()

for (i in 1:length(precip_files)){
  
  df = read.csv(paste0(temp_dir, precip_files[i]), header=FALSE)
  df = as.data.frame(t(df))
  
  n<-dim(df)[1]
  names <- df[(n-8),]
  colnames(df) <- names
  df <- df[1:(n-12),]
  
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
  mutate(date = lubridate::as_date(paste0(year, month, '01'), format='%Y%m%d'))%>%
  mutate(month = as.numeric(month), year = as.numeric(year))

df_precip_avg <- df_precip_all %>%
  group_by(month, Region) %>%
  summarise(month_avg_precip = mean(sum_precip))

df_precip_all <- df_precip_all %>%
  left_join(df_precip_avg, by=c('Region', 'month'))

plt_precip <- ggplot(df_precip_all)+
  geom_bar(aes(x=date, y=sum_precip),fill='lightblue', stat='identity')+
  geom_line(aes(x=date, y=month_avg_precip), color='darkred', size=0.25)+
  facet_grid(rows=vars(Region))+
  theme_bw()+
  labs(x='Date', y='Precipitation (mm)')+
  theme(legend.position = 'bottom')

# Get average monthly precip in each season
df_precip_season <- df_precip_all %>%
  as.data.frame()%>%
  mutate(season_approx = ifelse(df_precip_all$month >= 10, df_precip_all$year, ifelse(df_precip_all$month <= 7, df_precip_all$year - 1, 'outside rainy season')))%>%
  filter(season_approx!='outside rainy season') %>%
  group_by(Region, season_approx)%>%
  summarise(total_season_precip = sum(sum_precip))%>%
  mutate(season_approx = as.numeric(season_approx))

# # Compare precip and temp -----------------------------------------------

df_precip_temp <- df_precip_all %>%
  full_join(df_temp_all, by=c('Region', 'date'))

plt_precip_temp <- ggplot(df_precip_temp, aes(x=sum_precip, y=Temp, group=Region))+
  geom_point(aes(color=Region))+
  theme_minimal()+
  labs(x='Total monthly precipitation (mm)', y='Average monthly temperature (C)')


# Historical dry spells ---------------------------------------------------

# Aggregate this data to get the total number of dry spells per region
# in each growing season. Pretty much all the dry spells are in the Southern region.

df_dryspells_agg <- df_dryspells %>%
  group_by(region, season_approx) %>%
  summarise(num_ds= n()) %>%
  as.data.frame() %>%
  add_row(region = 'Northern', season_approx = 2000, num_ds=0) %>%
  complete(region, season_approx = 2000:2020, 
           fill = list(num_ds = 0)) 

# What about the total number of dry spell days in each region per growing season?
df_dryspells_days <- df_dryspells %>%
  group_by(region, season_approx)%>%
  summarise(days_ds=sum(dry_spell_duration)) %>%
  as.data.frame() %>%
  add_row(region = 'Northern', season_approx = 2000, days_ds=0) %>%
  complete(region, season_approx = 2000:2020, 
           fill = list(days_ds = 0)) 


# Historical dry spells - pixel-based -------------------------------------

df_dryspells_px <- df_dryspells_px %>%
  mutate(date= as.Date(date))%>%
  mutate(year = lubridate::year(date))%>%
  mutate(month_day = format(date, "%m-%d"))%>%
  mutate(date_no_year = as.Date(paste0('1800-',month_day), "%Y-%m-%d"))

plt_ds_px <- df_dryspells_px %>%
  ggplot(aes(x=date_no_year, y=perc_ds_cells, group=ADM1_EN))+
  geom_line(aes(color=ADM1_EN))+
  facet_wrap(~year)+
  theme_bw()+
  theme(legend.position = 'bottom')+
  scale_x_date(date_labels = "%b")+
  labs(x='Date', y='Percent of pixels in a dry spell', color='Region')

# Get the max dry spell fraction by season by adm1
df_ds_px_season <- df_dryspells_px %>%
  mutate(month = lubridate::month(date)) %>%
  mutate(season_approx = ifelse(df_dryspells_px$month >= 10, df_dryspells_px$year, ifelse(df_dryspells_px$month <= 7, df_dryspells_px$year - 1, 'outside rainy season')))%>%
  filter(season_approx!='outside rainy season') %>%
  dplyr::select('ADM1_EN', 'perc_ds_cells', 'season_approx') %>%
  group_by(ADM1_EN, season_approx) %>%
  summarise(max_ds_perc = max(perc_ds_cells))


# Understanding relationships ---------------------------------------------

# Join and get aggregate statistics by season by region
df_sum <- df_dryspells_days %>%
  full_join(df_dryspells_agg, by=c('region', 'season_approx'))%>%
  full_join(df_asi_sel_max, by=c('region'='Province', 'season_approx'='season_approx'))%>%
  full_join(df_fewsnet_season, by=c('region'='ADMIN1', 'season_approx'='season_approx'))%>%
  full_join(df_ds_px_season, by=c('region'='ADM1_EN', 'season_approx'='season_approx')) %>%
  full_join(df_wrsi_season, by=c('region'='ID', 'season_approx'='season_approx'))%>%
  full_join(df_temp_season, by=c('region'='Region', 'season_approx'='season_approx'))%>%
  full_join(df_precip_season, by=c('region'='Region', 'season_approx'='season_approx'))%>%
  filter(season_approx != 2020)


# Create scatter plots to understand relationships 
plot_scatter <- function(x, y, xlab, ylab){
  plt <- ggplot(df_sum, aes(y=y, x=x, color=region))+
    geom_point(alpha=0.5, size=2)+
    theme_minimal()+
    labs(x=xlab, y=ylab)
  return(plt)
}

group_cor <- function(grp, df){
  df_sel <- df %>%
    filter(region==grp)
  return(cor(df_sel[,5:8], use='p'))
}

cor_central <- group_cor('Central', df_sum)
cor_northern <- group_cor('Northern', df_sum)
cor_southern <- group_cor('Southern', df_sum)
cor_all_c <- cor(df_sum[,5:10], use='complete.obs')
cor_all_p <- cor(df_sum[,5:10], use='p')

plt_asi_mean_days <- plot_scatter(df_sum$days_ds, df_sum$avg_asi, 'Number of dry spell days', 'Mean ASI')
plt_asi_max_days <- plot_scatter(df_sum$days_ds, df_sum$max_asi, 'Number of dry spell days', 'Max ASI')

plt_ipc_days <- plot_scatter(df_sum$days_ds, df_sum$tot, 'Number of dry spell days', 'Population IPC 3+')
plt_ipc_asi <- plot_scatter(df_sum$avg_asi, df_sum$tot, 'Mean ASI', 'Population IPC 3+')


plot_scatter(df_sum$avg_season_temp, df_sum$total_season_precip, '', '')
plot_scatter(df_sum$max_ds_perc, df_sum$min_wrsi, '', '')

# Make corr plots ---------------------------------------------------------

p.mat <- cor_pmat(df_sum[,5:10])


plt_cor_all <- ggcorrplot(cor_all_c, 
                              #type = "lower",
                              p.mat = p.mat,
                              hc.order = TRUE,
                              insig = "blank", 
                              lab=TRUE,
                              sig.level = 0.05,
                              ggtheme=theme_bw())
plt_cor_all
