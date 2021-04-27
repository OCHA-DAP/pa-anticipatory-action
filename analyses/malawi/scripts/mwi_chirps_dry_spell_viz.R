library(sf)
library(ggplot2)
library(tmap)
library(dplyr)
library(RcppRoll)
library(tidyr)
library(glue)

theme_set(theme_bw())

# Setting path directories -----------------------------------------------------
data_dir <- Sys.getenv("AA_DATA_DIR")
dry_spell_dir <- paste0(data_dir, '/processed/malawi/dry_spells/')
exploration_dry_spell_dir <- paste0(data_dir,'/exploration/malawi/dryspells/')
data_mean_long <- readRDS(paste0(data_dir, "/processed/malawi/dry_spells/data_mean_values_long.RDS"))# Fill in rainy or dry spell dates
# df_dry_spells <- read.csv(paste0(dry_spell_dir, 'daily_mean_dry_spells_details_2mm_2000_2020.csv')) # 14+ consecutive days with <= 2mm rain

# Transform the data for plotting -----------------------------------------

fill_dates <- function(df_dates, df_rainy_season,start_col, end_col, fill_num){
  adms <- unique(df_rainy_season[['pcode']])
  dates <- seq(as.Date("2000-01-01"), as.Date("2020-12-31"), by="days")
  df <- data.frame(dates)
  for (adm in adms){
    df[adm] <- 0
    for(i in 1:nrow(df_dates)){
      if (df_dates$pcode[i] == adm){
        ds_start <- df_dates[[start_col]][i]
        ds_end <- df_dates[[end_col]][i]
        df[[adm]][(df$date > ds_start) & (df$date <= ds_end)] <- fill_num
      }
    }
  }
  # Clean up the df to a tidy format for ggplot
  df_long <- df %>%
    gather(pcode, day_type, head(adms,1):tail(adms,1))%>%
    mutate(day_fac = as.factor(day_type))%>%
    mutate(month_day = format(dates, "%m-%d"))%>%
    mutate(date_no_year = as.Date(paste0('1800-',month_day), "%Y-%m-%d"))%>%
    mutate(year = lubridate::year(dates))%>%
    mutate(region = substr(pcode, 3, 3)) %>%
    mutate(region = ifelse(region == 3, "Southern", ifelse(region == 2, "Central", "Northern")))

  return(df_long)
}

#part of function fill_dates, used if data already in semi-processed format
prepare_ggplot <- function(df,ds_col){
  df$dates <- as.Date(df$date,format="%Y-%m-%d")
  df$day_type <- df[[ds_col]]
  df_long <- df %>%
    mutate(day_fac = as.factor(day_type))%>%
    mutate(month_day = format(dates, "%m-%d"))%>%
    mutate(date_no_year = as.Date(paste0('1800-',month_day), "%Y-%m-%d"))%>%
    mutate(year = lubridate::year(dates)) %>%
    mutate(region = substr(pcode, 3, 3)) %>%
    mutate(region = ifelse(region == 3, "Southern", ifelse(region == 2, "Central", "Northern")))
  return(df_long)
}



plot_heatmap <- function(df_dry_spells,df_rainy_season, match_values,match_labels,color_scale,y_label,plot_title,output_path,ds_flatdata=FALSE,sub_title=""){
  #ds_flatdata: choose the appropriate preprocessing, depending on the format the data comes in
  #the output from mwi_chirps_dry_spell_detection.R should use ds_flatdata=FALSE
  #however, some cases the data is preprocessed in python in which case ds_flatdata=TRUE
  if (ds_flatdata) {
    df_ds <- prepare_ggplot(df_dry_spells,"dryspell_match")
  }
  else {
      df_ds <- fill_dates(df_dry_spells, df_rainy_season,'dry_spell_first_date', 'dry_spell_last_date', 1)
  }
  df_rs <- fill_dates(df_rainy_season, df_rainy_season,'onset_date', 'cessation_date', 10)

  df_ds %>%
    full_join(df_rs, by=c('pcode', 'dates'))%>%
    mutate(days = as.factor(day_type.x + day_type.y))%>%
    mutate(days = factor(days, levels=dry_spell_match_values, labels=match_values_labels))%>%
    drop_na(date_no_year.x) %>%
    arrange(desc(pcode),date_no_year.x) %>%
    ggplot(aes(x=date_no_year.x, y=pcode, fill=days))+
    geom_tile() +
    scale_fill_manual(values=color_scale)+
    facet_grid(rows=vars(year.x))+
    theme_minimal()+
    labs(title=plot_title, subtitle=sub_title,x='Date', y=y_label, fill='')+
    theme(axis.text.y=element_blank(),
          axis.ticks.y=element_blank(),
          legend.position = 'bottom',
          strip.text = element_text(size=16,angle=0),
          axis.text.x = element_text(size=16),
          legend.text = element_text(size=16),
          axis.title.x = element_text(size=16),
          axis.title.y = element_text(size=16),
          plot.title = element_text(size=32))+
    scale_x_date(date_labels = "%b")
  #uncomment to save plots
  # ggsave(output_path_hm,width=20,height=15)
}



# Make the heatmaps ------------------------------------------

# ## Occurrences of dry spells with different aggregation methodologies on adm2
#Set general variables for heatmap
dry_spell_match_values=c(10, 0, 11, 1)
match_values_labels=c("Rainy season", "Dry season", 'Dry spell in rainy season', "Dry spell in dry season")
color_scale=c('#b3e7ff', '#fff2d6',"#b52722",  '#fc8d5d')
y_label="Admin 2 region"

## mean
df_dry_spells <- read.csv(paste0(dry_spell_dir,'dry_spells_during_rainy_season_list_2000_2020_mean_back.csv'))
df_rainy_season <- read.csv(paste0(dry_spell_dir, "rainy_seasons_detail_2000_2020_mean_back.csv"))
plot_title=glue("20 years of dry spells in Malawi's admin 2 regions, 2000-2020") #not indicating aggregation method as this is our standard
output_path_hm=paste0(exploration_dry_spell_dir, glue('mwi_viz_hm_dry_spell_mean_adm2.png'))
plot_heatmap(df_dry_spells,df_rainy_season, dry_spell_match_values,match_values_labels,color_scale,y_label,plot_title,output_path_hm)

## max
df_dry_spells <- read.csv(paste0(dry_spell_dir,'dry_spells_during_rainy_season_list_2000_2020.csv'))
df_rainy_season <- read.csv(paste0(dry_spell_dir, "rainy_seasons_detail_2000_2020.csv"))
plot_title=glue("20 years of dry spells in Malawi's admin 2 regions, 2000-2020")
sub_title = "Dry spell determined by the cell with maximum precipitation per admin2"
output_path_hm=paste0(exploration_dry_spell_dir, glue('mwi_viz_hm_dry_spell_max_adm2.png'))
plot_heatmap(df_dry_spells,df_rainy_season, dry_spell_match_values,match_values_labels,color_scale,y_label,plot_title,output_path_hm,sub_title=sub_title)

## pixel-based
for (threshold in c(10,30,50,70,90)){
  df_dry_spells <- read.csv(paste0(exploration_dry_spell_dir,glue('dryspells_pixel_adm2_th{threshold}_viz.csv')))
  df_rainy_season <- read.csv(paste0(dry_spell_dir, "rainy_seasons_detail_2000_2020_per_pixel_adm2.csv"))
  df_rainy_season <- df_rainy_season %>%
    rename(pcode = ADM2_PCODE)
  plot_title=glue("20 years of dry spells in Malawi's admin 2 regions, 2000-2020")
  sub_title = glue("Dry spell in admin 2 is: >= {threshold}% cells within admin 2 had <=2mm precipitation during 14 cumulative days")
  output_path_hm=paste0(exploration_dry_spell_dir, glue('mwi_viz_hm_dry_spell_th_{threshold}_adm2.png'))
  plot_heatmap(df_dry_spells,df_rainy_season, dry_spell_match_values,match_values_labels,color_scale,y_label,plot_title,output_path_hm,ds_flatdata=TRUE,sub_title=sub_title)

}

## ADMIN3 mean
y_label="Admin 3 region"
df_dry_spells <- read.csv(paste0(dry_spell_dir, 'dry_spells_during_rainy_season_list_2000_2020_adm3.csv'))
df_rainy_season <- read.csv(paste0(dry_spell_dir, 'rainy_seasons_detail_2000_2020_ADM3.csv'))
plot_title=glue("20 years of dry spells in Malawi's admin 3 regions, 2000-2020")
output_path_hm=paste0(exploration_dry_spell_dir, glue('mwi_viz_hm_dry_spell_mean_adm3.png'))
plot_heatmap(df_dry_spells,df_rainy_season, dry_spell_match_values,match_values_labels,color_scale,y_label,plot_title,output_path_hm)

## ADMIN1 pixel-based
#define file paths
threshold <- 50
df_dry_spells <- read.csv(paste0(exploration_dry_spell_dir,glue('dryspells_pixel_adm1_th{threshold}_viz.csv')))
df_rainy_season <- read.csv(paste0(exploration_dry_spell_dir, "rainy_seasons_detail_2000_2020_per_pixel_adm1.csv"))


#TODO: create plots for the less than xmm per day methodology
#ggsave(paste0(dry_spell_dir, '/dry_spell_plots/mean_back_dry_spell_hm.png'))
#ggsave(paste0(dry_spell_dir, '/dry_spell_plots/dry_spell_hm_consecutive_days_4mm.png'), width = 7.55, height = 7.82, units = "in", dpi = 300)
#ggsave(paste0(dry_spell_dir, '/dry_spell_plots/dry_spell_hm_consecutive_days_2mm.png'), width = 7.55, height = 7.82, units = "in", dpi = 300)


#Set variables for heatmap
y_label="Admin 1 region"
plot_title=glue('Observed dry spells by CHIRPS pixel-based aggregation on ADM1, {threshold}% of pixels threshold')
output_path_hm=paste0(exploration_dry_spell_dir, glue('viz_chirps_pixel_adm1_th_{threshold}_dryspell_hm_test.png'))
ds_flatdata=TRUE
# #TODO: add option to add yticks labels
# yticks_text=TRUE
plot_heatmap(df_dry_spells,df_rainy_season, dry_spell_match_values,match_values_labels,color_scale,y_label,plot_title,output_path_hm,ds_flatdata,yticks_text)

### Make heatmap of the match of observed dry spells with different aggregation methodologies and different indicators
## Mean vs >=50% pixels
#Set general variables for heatmap
dry_spell_match_values=c(10, 0, 13,12,11)
match_values_labels=c("Rainy season", "Dry season", glue('mean and >={threshold} dry spell'),'mean dry spell',glue('>={threshold} dry spell'))
color_scale=c('#b3e7ff', '#fff2d6','#0063B3' ,'#F2645A',"#78D9D1")
y_label="Admin 2 region"

threshold=50
df_dry_spells <- read.csv(paste0(exploration_dry_spell_dir,glue('dryspells_mean_pixel_th{threshold}_viz.csv')))
df_rainy_season <- read.csv(paste0(dry_spell_dir, "rainy_seasons_detail_2000_2020_mean_back.csv"))
plot_title=glue('Observed dry spells per admin 2 with mean and >={threshold}% aggregation methodology')
output_path_hm=paste0(exploration_dry_spell_dir, glue('mwi_viz_hm_dry_spell_mean_th{threshold}_adm2.png'))
plot_heatmap(df_dry_spells,df_rainy_season, dry_spell_match_values,match_values_labels,color_scale,y_label,plot_title,output_path_hm,ds_flatdata=TRUE)

## CHIRPS-GEFS
#Set general variables for heatmap
dry_spell_match_values=c(10, 0, 13,12,11)
match_values_labels=c("Rainy season", "Dry season", 'Observed and forecasted dry spell','Observed dry spell','Forecasted dry spell')
color_scale=c('#b3e7ff', '#fff2d6','#0063B3' ,'#F2645A',"#78D9D1")
y_label="Admin 2 region"

for (threshold in c(2,10,20,25)) {
  df_dry_spells <- read.csv(paste0(dry_spell_dir,glue('chirpsgefs/dryspells_chirpsgefs_dates_viz_mean_2mm_th{threshold}.csv')))
  df_rainy_season <- read.csv(paste0(dry_spell_dir, "rainy_seasons_detail_2000_2020_mean_back.csv"))
  plot_title=glue("Overlap observed and forecasted dry spells by CHIRPS-GEFS") #not indicating aggregation method as this is our standard
  sub_title=glue("CHIRPS-GEFS dry spell if 15-day cumulative precipitation <={threshold} mm")
  output_path_hm=paste0(exploration_dry_spell_dir, glue('mwi_viz_hm_chirps_gefs_mean_th{threshold}_adm2.png'))
  plot_heatmap(df_dry_spells,df_rainy_season, dry_spell_match_values,match_values_labels,color_scale,y_label,plot_title,output_path_hm,ds_flatdata=TRUE,sub_title=sub_title)
}





# Line plot
# implemented for computing historical dry spells with mean aggregation on admin2
df_dry_spells_mean <- read.csv(paste0(dry_spell_dir, 'daily_mean_dry_spells_details_2mm_2000_2020.csv'))
df_ds <- fill_dates(df_dry_spells_mean, 'dry_spell_first_date', 'dry_spell_last_date', 1)
df_ds %>%
  mutate(region = as.factor(region))%>%
  mutate(region = forcats::fct_relevel(region, 'Northern', 'Central', 'Southern'))%>%
  group_by(dates, region)%>%
  summarize(count_spells = sum(day_type))%>%
  mutate(month_day = format(dates, "%m-%d"))%>%
  mutate(date_no_year = as.Date(paste0('1800-',month_day), "%Y-%m-%d"))%>%
  mutate(year = lubridate::year(dates))%>%
  ggplot(aes(x=date_no_year, y=count_spells, group=year))+
  geom_line(aes(color=year))+
  facet_grid(rows=vars(region))+
  scale_x_date(date_labels = "%b")+
  theme_minimal()+
  labs(title='Number of dry spells by region in Malawi, 2000-2020',
       x='Date',
       y='Count')

#ggsave(paste0(dry_spell_dir, '/dry_spell_plots/mean_back_dry_spell_line.png'))
#ggsave(paste0(dry_spell_dir, '/dry_spell_plots/dry_spell_line_consec_days_4mm.png'), width = 7.55, height = 7.82, units = "in", dpi = 300)
ggsave(paste0(dry_spell_dir, '/dry_spell_plots/dry_spell_line_consec_days_2mm.png'), width = 7.55, height = 7.82, units = "in", dpi = 300)



# ARCHIVE - NOT NEEDED ----------------------------------------------------


# Combine the max and mean data -------------------------------------------

df_dry_spells_max$type <- 'MAX'
df_dry_spells_mean$type <- 'MEAN'
df_dry_spells <- rbind(df_dry_spells_max, df_dry_spells_mean)


# Basic summary stats -----------------------------------------------------

# What is the average dry spell duration?
mean(df_dry_spells_max$dry_spell_duration)
mean(df_dry_spells_mean$dry_spell_duration)

# What is the total number of dry spells?
nrow(df_dry_spells_max)
nrow(df_dry_spells_max_mean)

# Are there any cases of multiple dry spells within a single season in the same area?
df_dry_spells %>%
  group_by(ADM2_EN, season_approx, type)%>%
  summarise(n()) %>%
  filter(`n()`>1)


# Making some basic plots -------------------------------------------------

# What is the average duration of dry spells per season?
df_dry_spells %>%
  group_by(season_approx, type)%>%
  summarize(avg = mean(dry_spell_duration))%>%
  ggplot(aes(x=season_approx, y=avg, fill=type))+
  geom_bar(stat='identity', position='dodge') +
  coord_flip()+
  labs(y='Duration (days)',
       title = 'Average dry spell duration by season',
       x='Season',
       fill='Aggregation\ntype')
ggsave('Avg_duration_by_season.png')

# What is the average duration of dry spells per adm 2 region?
df_dry_spells %>%
  group_by(ADM2_EN, type)%>%
  summarize(avg = mean(dry_spell_duration))%>%
  ggplot(aes(x=ADM2_EN, y=avg, fill=type))+
  geom_bar(stat='identity', position='dodge') +
  coord_flip() +
  labs(y='Duration (days)',
       title = 'Average dry spell duration by admin 2 region',
       x='Admin 2 region',
       fill='Aggregation\ntype')
ggsave('Avg_duration_by_adm2.png')

# What is the number of dry spells per season?
df_dry_spells %>%
  group_by(season_approx, type)%>%
  summarize(n())%>%
  ggplot(aes(x=season_approx, y=`n()`, fill=type))+
  geom_bar(stat='identity', position='dodge') +
  coord_flip()+
  labs(y='Count',
       title = 'Number of dry spells by season',
       x='Season',
       fill='Aggregation\ntype')
ggsave('Count_by_season.png')

# What is the number of dry spells per admin 2 region?
df_dry_spells %>%
  group_by(ADM2_EN, type)%>%
  summarise(n())%>%
  ggplot(aes(x=ADM2_EN, y=`n()`, fill=type)) +
  geom_bar(stat = 'identity', position='dodge') +
  coord_flip() +
  labs(title = 'Total dry spells per admin 2 region',
       y='Count',
       x='Admin 2 region',
       fill='Aggregation\ntype')
ggsave('Count_by_adm2.png')

cat <- 'MAX'

# For each season, what is the length of each dry spell in each admin region?
df_dry_spells %>%
  filter(type == cat) %>%
  ggplot(aes(x=ADM2_EN, y=dry_spell_duration, fill=dry_spell_rainfall)) +
  geom_bar(stat = 'identity') +
  #scale_fill_gradient(low = "red", high = "yellow")+
  coord_flip() +
  facet_wrap(~season_approx) +
  labs(title = paste0(cat, ': Duration of dry spells in each admin 2 region by season'),
       x='Admin 2 region',
       y='Duration (days)',
       fill='Total rainfall (mm)')+
  theme(axis.text.y=element_blank(),
        axis.ticks.y=element_blank())
 ggsave(paste0(dry_spell_dir, cat, '_duration_facet_rainfall.png'))


# Heatmaps to show time series patterns -----------------------------------
# Too much data to really see what's going on here...


# What are the precipitation values over time?
data_mean_long %>%
  mutate(month_day = format(date, "%m-%d"))%>%
  ggplot(aes(x=month_day, y=pcode, fill=total_prec))+
  geom_tile() +
  scale_fill_gradient(low = "white", high = "red")+
  facet_grid(rows=vars(year))+
   theme(axis.text.y=element_blank(),
         axis.ticks.y=element_blank(),
         axis.text.x=element_blank(),
         axis.ticks.x=element_blank())

# When is the 14-day rolling sum < 2mm?
data_mean_long %>%
  filter(year == 2000) %>%
  mutate(month_day = format(date, "%m-%d"))%>%
  #ggplot(aes(x=month_day, y=pcode, fill=rollsum_14d_less_than_2_bin))+
  ggplot(aes(x=month_day, y=pcode, fill=total_prec))+
  geom_tile() +
  scale_fill_gradient(low = "white", high = "red")+
  facet_grid(rows=vars(year))+
  theme(axis.text.y=element_blank(),
       axis.ticks.y=element_blank(),
       axis.text.x=element_blank(),
       axis.ticks.x=element_blank())


# --- Heatmap to show the occurrence of dry spells by admin 2 region
# --- throughout the calendar year
plot_rainy <- data_mean_rainy %>%
  filter(pcode == 'MW107')%>%
  ggplot(aes(x=date_no_year, y=pcode, fill=dry_fac))+
  scale_fill_manual(values=c("#ade0ff", "#f53b0c"))+
  facet_grid(rows=vars(year))+
  geom_tile()+
  theme_minimal()+
  labs(title='20 years of daily precipitation during rainy seasons in Malawi\'s Admin 2 regions, 2000-2020', x='Date', y='Admin 2 region', fill='Precipitation (mm)')+
  theme(axis.text.y=element_blank(),
        axis.ticks.y=element_blank())+
  scale_x_date(date_labels = "%b")
#plot_rainy + scale_fill_gradient(trans = "pseudo_log")
plot_rainy
ggsave('precipitation.png')

# -- Line graph with number of co-occurring dry spells for a given date
# Color lines by region
# Facet by year
# Need for each day and region, need to get the total number of dry spells
data_line_rainy <- data_mean_long %>%
  mutate(region = substr(pcode, 3, 3)) %>%
  mutate(region = ifelse(region == 3, "Southern", ifelse(region == 2, "Central", "Northern")))%>%
  group_by(date, region)%>%
  summarize(count_spells = sum(rollsum_14d_less_than_2_bin))%>%
  mutate(year = lubridate::year(date))%>%
  mutate(month_day = format(date, "%m-%d"))%>%
  # Bit hacky to get all the dates overlapping between years
  mutate(date_no_year = as.Date(paste0('1800-',month_day), "%Y-%m-%d"))%>%
  mutate(year_fac = as.factor(year))

plot_line <- data_line_rainy %>%
  ggplot(aes(x=date_no_year, y=count_spells, group=year_fac))+
  geom_line(aes(color=year_fac))+
  facet_grid(rows=vars(region))+
  scale_x_date(date_labels = "%b")
plot_line
