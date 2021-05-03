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
    scale_x_date(date_labels = "%b",date_breaks = "1 month",expand=c(0,0))
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


### Below threshold monthly precipitation and occurrence of dry spells
#Set general variables for heatmap
threshold=80
#classify below threshold monthly precip during dry season as dry season
dry_spell_match_values=c(10, 0, 1, 13,12,11)
match_values_labels=c("Rainy season", "Dry season","Dry season", glue('Observed dry spell and <={threshold} mm monthly precipitation'),'Observed dry spell',glue('<={threshold} mm monthly precipitation'))
color_scale=c('#b3e7ff', '#fff2d6','#0063B3' ,'#F2645A',"#78D9D1")
y_label="Admin 2 region"
df_dry_spells <- read.csv(paste0(dry_spell_dir,'seasonal/',glue('monthly_dryspellobs_th{threshold}.csv')))
df_rainy_season <- read.csv(paste0(dry_spell_dir, "rainy_seasons_detail_2000_2020_mean_back.csv"))
plot_title=glue("Overlap observed dry spells and <={threshold} mm monthly precipitation")
output_path_hm=paste0(exploration_dry_spell_dir, glue('mwi_viz_hm_dry_spell_monthly_precip_mean_th{threshold}_adm2.png'))
plot_heatmap(df_dry_spells,df_rainy_season, dry_spell_match_values,match_values_labels,color_scale,y_label,plot_title,output_path_hm,ds_flatdata=TRUE)

### only southern: Below threshold monthly precipitation and occurrence of dry spells
#Set general variables for heatmap
threshold=110
#classify match dry spell and below threshold precip during dry season as dry season (occurred once on 16-03-2020)
dry_spell_match_values=c(10, 0, 1,3, 13,12,11)
match_values_labels=c("Rainy season", "Dry season","Dry season", "Dry season",glue('Observed dry spell and <={threshold} mm monthly precipitation'),'Observed dry spell',glue('<={threshold} mm monthly precipitation'))
color_scale=c('#b3e7ff', '#fff2d6','#0063B3' ,'#F2645A',"#78D9D1")
y_label="Admin 2 district within the Southern region"
df_dry_spells <- read.csv(paste0(dry_spell_dir,'seasonal/',glue('monthly_dryspellobs_th{threshold}_southern.csv')))
df_rainy_season <- read.csv(paste0(dry_spell_dir, "rainy_seasons_detail_2000_2020_mean_back.csv"))
#only select adm2's within the southern region
df_rainy_season_southern <- df_rainy_season %>% mutate(region = substr(pcode, 3, 3)) %>%  filter(region==3)
plot_title=glue("Overlap observed dry spells and <={threshold} mm monthly precipitation for the Southern region")
output_path_hm=paste0(exploration_dry_spell_dir, glue('mwi_viz_hm_dry_spell_monthly_precip_mean_th{threshold}_adm2_southern.png'))
plot_heatmap(df_dry_spells,df_rainy_season_southern, dry_spell_match_values,match_values_labels,color_scale,y_label,plot_title,output_path_hm,ds_flatdata=TRUE)



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