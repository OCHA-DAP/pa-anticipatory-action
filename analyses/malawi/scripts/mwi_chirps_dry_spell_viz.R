library(sf)
library(ggplot2)
library(tmap)
library(dplyr)
library(RcppRoll)
library(tidyr)
library(glue)
library(flextable)

theme_set(theme_bw())

# load functions
source("scripts/mwi_chirps_dry_spell_viz_hm_functions.R")

# Setting path directories -----------------------------------------------------
data_dir <- Sys.getenv("AA_DATA_DIR")
dry_spell_dir <- paste0(data_dir, '/public/processed/mwi/dry_spells/')
exploration_dry_spell_dir <- paste0(data_dir,'/public/exploration/mwi/dryspells/')
data_mean_long <- readRDS(paste0(data_dir, "/public/processed/mwi/dry_spells/data_mean_values_long.RDS"))# Fill in rainy or dry spell dates



# Make the heatmaps ------------------------------------------

# # ## Occurrences of dry spells with different aggregation methodologies on adm2
### Different aggregations for cumul <=2mm
# #Set general variables for heatmap
# dry_spell_match_values=c(10, 0, 11, 1)
# match_values_labels=c("Rainy season", "Dry season", 'Dry spell in rainy season', "Dry spell in dry season")
# color_scale=c('#b3e7ff', '#fff2d6',"#b52722",  '#fc8d5d')
# y_label="Admin 2 region"
# 
# ## mean 
# df_dry_spells <- read.csv(paste0(dry_spell_dir,'dry_spells_during_rainy_season_list_2000_2020_mean_back.csv'))
# df_rainy_season <- read.csv(paste0(dry_spell_dir, "rainy_seasons_detail_2000_2020_mean_back.csv"))
# plot_title=glue("20 years of dry spells in Malawi's admin 2 regions, 2000-2020") #not indicating aggregation method as this is our standard
# output_path_hm=paste0(exploration_dry_spell_dir, glue('mwi_viz_hm_dry_spell_mean_adm2.png'))
# hm_mean <- plot_heatmap(df_dry_spells,df_rainy_season, dry_spell_match_values,match_values_labels,color_scale,y_label,plot_title,output_path_hm)
# # ggsave(output_path_hm,plot = hm_mean, width=20,height=15)
# 
# ## max
# df_dry_spells <- read.csv(paste0(dry_spell_dir,'dry_spells_during_rainy_season_list_2000_2020.csv'))
# df_rainy_season <- read.csv(paste0(dry_spell_dir, "rainy_seasons_detail_2000_2020.csv"))
# plot_title=glue("20 years of dry spells in Malawi's admin 2 regions, 2000-2020")
# sub_title = "Dry spell determined by the cell with maximum precipitation per admin2"
# output_path_hm=paste0(exploration_dry_spell_dir, glue('mwi_viz_hm_dry_spell_max_adm2.png'))
# hm_max <- plot_heatmap(df_dry_spells,df_rainy_season, dry_spell_match_values,match_values_labels,color_scale,y_label,plot_title,output_path_hm,sub_title=sub_title)
# # ggsave(output_path_hm,plot = hm_max, width=20,height=15)
# 
# ## pixel-based
# for (threshold in c(10,30,50,70,90)){
#   df_dry_spells <- read.csv(paste0(exploration_dry_spell_dir,glue('dryspells_pixel_adm2_th{threshold}_viz.csv')))
#   df_rainy_season <- read.csv(paste0(dry_spell_dir, "rainy_seasons_detail_2000_2020_per_pixel_adm2.csv"))
#   df_rainy_season <- df_rainy_season %>%
#     rename(pcode = ADM2_PCODE)
#   plot_title=glue("20 years of dry spells in Malawi's admin 2 regions, 2000-2020")
#   sub_title = glue("Dry spell in admin 2 is: >= {threshold}% cells within admin 2 had <=2mm precipitation during 14 cumulative days")
#   output_path_hm=paste0(exploration_dry_spell_dir, glue('mwi_viz_hm_dry_spell_th_{threshold}_adm2.png'))
#   hm_pixel <- plot_heatmap(df_dry_spells,df_rainy_season, dry_spell_match_values,match_values_labels,color_scale,y_label,plot_title,output_path_hm,ds_flatdata=TRUE,sub_title=sub_title)
#   # ggsave(output_path_hm,plot = hm_pixel, width=20,height=15)
# }
# 
# ## ADMIN3 mean
# y_label="Admin 3 region"
# df_dry_spells <- read.csv(paste0(dry_spell_dir, 'dry_spells_during_rainy_season_list_2000_2020_adm3.csv'))
# df_rainy_season <- read.csv(paste0(dry_spell_dir, 'rainy_seasons_detail_2000_2020_ADM3.csv'))
# plot_title=glue("20 years of dry spells in Malawi's admin 3 regions, 2000-2020")
# output_path_hm=paste0(exploration_dry_spell_dir, glue('mwi_viz_hm_dry_spell_mean_adm3.png'))
# hm_adm3 <- plot_heatmap(df_dry_spells,df_rainy_season, dry_spell_match_values,match_values_labels,color_scale,y_label,plot_title,output_path_hm)
# # ggsave(output_path_hm,plot = hm_adm3, width=20,height=15)
# 

#this is not working properly yet.. 
## ADMIN1 pixel-based
# #define file paths
# threshold <- 30
# df_dry_spells <- read.csv(paste0(exploration_dry_spell_dir,glue('dryspells_pixel_adm1_th{threshold}_viz.csv')))
# df_rainy_season <- read.csv(paste0(exploration_dry_spell_dir, "rainy_seasons_detail_2000_2020_per_pixel_adm1.csv"))
# # flextable(df_rainy_season)
# flextable(df_dry_spells)
# #Set variables for heatmap
# dry_spell_match_values=c(10, 0, 11, 1)
# match_values_labels=c("Rainy season", "Dry season", 'Dry spell in rainy season', "Dry spell in dry season")
# color_scale=c('#b3e7ff', '#fff2d6',"#b52722",  '#fc8d5d')
# y_label="Admin 1 region"
# plot_title=glue('Observed dry spells by CHIRPS pixel-based aggregation on ADM1, {threshold}% of pixels threshold')
# output_path_hm=paste0(exploration_dry_spell_dir, glue('viz_chirps_pixel_adm1_th_{threshold}_dryspell_hm_test.png'))
# ds_flatdata=TRUE
# # #TODO: add option to add yticks labels
# # yticks_text=TRUE
# hm_adm1 <- plot_heatmap(df_dry_spells,df_rainy_season, dry_spell_match_values,match_values_labels,color_scale,y_label,plot_title,output_path_hm,ds_flatdata)#,yticks_text)
# hm_adm1
# ggsave(output_path_hm,plot = hm_adm1, width=20,height=15)
# 

# ### only southern in decjanfeb: Below threshold monthly precipitation and occurrence of dry spells
# #kind of ugly way of doing this, cause using a file where dry spell is compared with monthly precipitaiton.
# #but it was the quickest method..
# #Set general variables for heatmap
# 
# dry_spell_match_values=c(3,0,2,1)
# match_values_labels=c("Dry spell", "No dry spell","Dry spell","No dry spell")
# color_scale=c('#b52722','#b3e7ff','#b52722',"#b3e7ff")# ,) #,'#fff2d6'
# y_label="Admin 2 district"
# df_dry_spells <- read.csv(paste0(dry_spell_dir,'seasonal/',glue('monthly_dryspellobs_th100_southern_decjanfeb.csv')))
# df_rainy_season <- read.csv(paste0(dry_spell_dir, "rainy_seasons_detail_2000_2020_mean_back.csv"))
# #only select adm2's within the southern region
# df_rainy_season_southern <- df_rainy_season %>% mutate(region = substr(pcode, 3, 3)) %>%  filter(region==3)
# 
# plot_title=glue("Observed dry spells per adm2 for the Southern region in DecJanFeb")
# sub_title="The year corresponds to the start of the rainy season"
# output_path_hm=paste0(exploration_dry_spell_dir, glue('mwi_viz_hm_dry_spell_mean_adm2_southern_decjanfeb.png'))
# hm_monthly_southern_sel <-plot_heatmap_without_rainy(df_dry_spells,df_rainy_season_southern, dry_spell_match_values,match_values_labels,color_scale,y_label,plot_title,output_path_hm,ds_flatdata=TRUE,sub_title=sub_title)
# hm_monthly_southern_sel
# ggsave(output_path_hm,plot = hm_monthly_southern_sel, width=20,height=15)

# ### Detect dry spell based on daily maximum precipitation instead of cumulative
# #Set general variables for heatmap
# dry_spell_match_values=c(10, 0, 11, 1)
# match_values_labels=c("Rainy season", "Dry season", 'Dry spell in rainy season', "Dry spell in dry season")
# color_scale=c('#b3e7ff', '#fff2d6',"#b52722",  '#fc8d5d')
# y_label="Admin 2 region"
# 
# #max 2mm/day
# df_dry_spells <- read.csv(paste0(dry_spell_dir, 'daily_mean_dry_spells_details_2mm_2000_2020.csv'))
# df_rainy_season <- read.csv(paste0(dry_spell_dir, "rainy_seasons_detail_2000_2020_mean_back.csv"))
# plot_title=glue("dry spells with <=2mm/day in Malawi's admin2 regions, 2000-2020")
# output_path_hm=paste0(exploration_dry_spell_dir, glue('mwi_viz_hm_dry_spell_2mm_daily_mean_adm2.png'))
# hm_2mm <- plot_heatmap(df_dry_spells,df_rainy_season, dry_spell_match_values,match_values_labels,color_scale,y_label,plot_title,output_path_hm)
# ggsave(output_path_hm,plot = hm_2mm, width=20,height=15)
# 
# #max 4mm/day
# df_dry_spells <- read.csv(paste0(dry_spell_dir, 'daily_mean_dry_spells_details_2000_2020.csv'))
# df_rainy_season <- read.csv(paste0(dry_spell_dir, "rainy_seasons_detail_2000_2020_mean_back.csv"))
# plot_title=glue("dry spells with <=4mm/day in Malawi's admin2 regions, 2000-2020")
# output_path_hm=paste0(exploration_dry_spell_dir, glue('mwi_viz_hm_dry_spell_4mm_daily_mean_adm2.png'))
# hm_4mm <- plot_heatmap(df_dry_spells,df_rainy_season, dry_spell_match_values,match_values_labels,color_scale,y_label,plot_title,output_path_hm)
# ggsave(output_path_hm,plot = hm_4mm, width=20,height=15)
# hm_4mm

# 
# 
# 
# ### Make heatmap of the match of observed dry spells with different aggregation methodologies and different indicators
# ## Mean vs >=50% pixels
# #Set general variables for heatmap
# dry_spell_match_values=c(10, 0, 13,12,11)
# match_values_labels=c("Rainy season", "Dry season", glue('mean and >={threshold} dry spell'),'mean dry spell',glue('>={threshold} dry spell'))
# color_scale=c('#b3e7ff', '#fff2d6','#0063B3' ,'#F2645A',"#78D9D1")
# y_label="Admin 2 region"
# 
# threshold=50
# df_dry_spells <- read.csv(paste0(exploration_dry_spell_dir,glue('dryspells_mean_pixel_th{threshold}_viz.csv')))
# df_rainy_season <- read.csv(paste0(dry_spell_dir, "rainy_seasons_detail_2000_2020_mean_back.csv"))
# plot_title=glue('Observed dry spells per admin 2 with mean and >={threshold}% aggregation methodology')
# output_path_hm=paste0(exploration_dry_spell_dir, glue('mwi_viz_hm_dry_spell_mean_th{threshold}_adm2.png'))
# hm_mean_pixel <- plot_heatmap(df_dry_spells,df_rainy_season, dry_spell_match_values,match_values_labels,color_scale,y_label,plot_title,output_path_hm,ds_flatdata=TRUE)
# # ggsave(output_path_hm,plot = hm_mean_pixel, width=20,height=15)
# 
# ## CHIRPS-GEFS
# #Set general variables for heatmap
# dry_spell_match_values=c(10, 0, 13,12,11)
# match_values_labels=c("Rainy season", "Dry season", 'Observed and forecasted dry spell','Observed dry spell','Forecasted dry spell')
# color_scale=c('#b3e7ff', '#fff2d6','#0063B3' ,'#F2645A',"#78D9D1")
# y_label="Admin 2 region"
# 
# for (threshold in c(2,10,20,25)) {
#   df_dry_spells <- read.csv(paste0(dry_spell_dir,glue('chirpsgefs/dryspells_chirpsgefs_dates_viz_mean_2mm_th{threshold}.csv')))
#   df_rainy_season <- read.csv(paste0(dry_spell_dir, "rainy_seasons_detail_2000_2020_mean_back.csv"))
#   plot_title=glue("Overlap observed and forecasted dry spells by CHIRPS-GEFS") #not indicating aggregation method as this is our standard
#   sub_title=glue("CHIRPS-GEFS dry spell if 15-day cumulative precipitation <={threshold} mm")
#   output_path_hm=paste0(exploration_dry_spell_dir, glue('mwi_viz_hm_chirps_gefs_mean_th{threshold}_adm2.png'))
#   hm_chirpsgefs <- plot_heatmap(df_dry_spells,df_rainy_season, dry_spell_match_values,match_values_labels,color_scale,y_label,plot_title,output_path_hm,ds_flatdata=TRUE,sub_title=sub_title)
#   # ggsave(output_path_hm,plot = hm_chirpsgefs, width=20,height=15)
#   }
# 
# 
# ### Below threshold monthly precipitation and occurrence of dry spells
# #Set general variables for heatmap
# threshold=80
# #classify below threshold monthly precip during dry season as dry season
# dry_spell_match_values=c(10, 0, 1, 13,12,11)
# match_values_labels=c("Rainy season", "Dry season","Dry season", glue('Observed dry spell and <={threshold} mm monthly precipitation'),'Observed dry spell',glue('<={threshold} mm monthly precipitation'))
# color_scale=c('#b3e7ff', '#fff2d6','#0063B3' ,'#F2645A',"#78D9D1")
# y_label="Admin 2 region"
# df_dry_spells <- read.csv(paste0(dry_spell_dir,'seasonal/',glue('monthly_dryspellobs_th{threshold}.csv')))
# df_rainy_season <- read.csv(paste0(dry_spell_dir, "rainy_seasons_detail_2000_2020_mean_back.csv"))
# plot_title=glue("Overlap observed dry spells and <={threshold} mm monthly precipitation")
# output_path_hm=paste0(exploration_dry_spell_dir, glue('mwi_viz_hm_dry_spell_monthly_precip_mean_th{threshold}_adm2.png'))
# hm_monthly <- plot_heatmap(df_dry_spells,df_rainy_season, dry_spell_match_values,match_values_labels,color_scale,y_label,plot_title,output_path_hm,ds_flatdata=TRUE)
# # ggsave(output_path_hm,plot = hm_monthly, width=20,height=15)
# 
# ### only southern: Below threshold monthly precipitation and occurrence of dry spells
# #Set general variables for heatmap
# threshold=110
# #classify match dry spell and below threshold precip during dry season as dry season (occurred once on 16-03-2020)
# dry_spell_match_values=c(10, 0, 1,3, 13,12,11)
# match_values_labels=c("Rainy season", "Dry season","Dry season", "Dry season",glue('Observed dry spell and <={threshold} mm monthly precipitation'),'Observed dry spell',glue('<={threshold} mm monthly precipitation'))
# color_scale=c('#b3e7ff', '#fff2d6','#0063B3' ,'#F2645A',"#78D9D1")
# y_label="Admin 2 district within the Southern region"
# df_dry_spells <- read.csv(paste0(dry_spell_dir,'seasonal/',glue('monthly_dryspellobs_th{threshold}_southern.csv')))
# df_rainy_season <- read.csv(paste0(dry_spell_dir, "rainy_seasons_detail_2000_2020_mean_back.csv"))
# #only select adm2's within the southern region
# df_rainy_season_southern <- df_rainy_season %>% mutate(region = substr(pcode, 3, 3)) %>%  filter(region==3)
# plot_title=glue("Overlap observed dry spells and <={threshold} mm monthly precipitation for the Southern region")
# output_path_hm=paste0(exploration_dry_spell_dir, glue('mwi_viz_hm_dry_spell_monthly_precip_mean_th{threshold}_adm2_southern.png'))
# hm_monthly_southern <-plot_heatmap(df_dry_spells,df_rainy_season_southern, dry_spell_match_values,match_values_labels,color_scale,y_label,plot_title,output_path_hm,ds_flatdata=TRUE)
# # ggsave(output_path_hm,plot = hm_monthly_southern, width=20,height=15)

# ### Only southern and Dec, Jan, Feb: Below threshold monthly precipitation and occurrence of dry spells
# #Set general variables for heatmap
# threshold=100
# #classify match dry spell and below threshold precip during dry season as dry season (occurred once on 16-03-2020)
# dry_spell_match_values=c(3,0,2,1)
# match_values_labels=c(glue("Observed dry spell and <={threshold} monthly precipitation"), glue("No observed dry spell and >{threshold} monthly precipitation"),"Observed dry spell",glue('<={threshold} mm monthly precipitation'))
# color_scale=c('#0063B3','#b3e7ff','#F2645A',"#78D9D1")# ,) #,'#fff2d6'
# y_label="Admin 2 district within the Southern region"
# df_dry_spells <- read.csv(paste0(dry_spell_dir,'seasonal/',glue('monthly_dryspellobs_th{threshold}_southern_decjanfeb.csv')))
# df_rainy_season <- read.csv(paste0(dry_spell_dir, "rainy_seasons_detail_2000_2020_mean_back.csv"))
# #only select adm2's within the southern region
# df_rainy_season_southern <- df_rainy_season %>% mutate(region = substr(pcode, 3, 3)) %>%  filter(region==3)
# # flextable(df_rainy_season_southern)
# plot_title=glue("Overlap observed dry spells and <={threshold} mm monthly precipitation for the Southern region")
# output_path_hm=paste0(exploration_dry_spell_dir, glue('mwi_viz_hm_dry_spell_monthly_precip_mean_th{threshold}_adm2_southern_decjanfeb.png'))
# hm_monthly_southern_sel <-plot_heatmap_without_rainy(df_dry_spells,df_rainy_season_southern, dry_spell_match_values,match_values_labels,color_scale,y_label,plot_title,output_path_hm,ds_flatdata=TRUE)
# ggsave(output_path_hm,plot = hm_monthly_southern_sel, width=20,height=15)
  


# ### Only southern and Dec, Jan, Feb: Below threshold monthly precipitation and occurrence of dry spells on ADM1
# #Set general variables for heatmap
# threshold<- 180
# #classify match dry spell and below threshold precip during dry season as dry season (occurred once on 16-03-2020)
# dry_spell_match_values=c(3,0,2,1)
# match_values_labels=c(glue("Observed dry spell and <={threshold} monthly precipitation"), glue("No observed dry spell and >{threshold} monthly precipitation"),"Observed dry spell",glue('<={threshold} mm monthly precipitation'))
# color_scale=c('#0063B3','#b3e7ff','#F2645A',"#78D9D1")# ,) #,'#fff2d6'
# y_label=""
# df_dry_spells <- read.csv(paste0(dry_spell_dir,'seasonal/',glue('monthly_dryspellobs_adm1_th{threshold}_southern_decjanfeb.csv')))
# df_rainy_season <- read.csv(paste0(dry_spell_dir, "rainy_seasons_detail_2000_2020_mean_back.csv"))
# #only  the southern region
# df_rainy_season_southern <- df_rainy_season %>% mutate(region = substr(pcode, 3, 3)) %>%  filter(region==3)
# # flextable(df_rainy_season_southern)
# plot_title=glue("Overlap observed dry spells and <={threshold} mm monthly precipitation for the Southern region")
# sub_title="The year corresponds to the start of the rainy season"
# output_path_hm=paste0(exploration_dry_spell_dir, glue('mwi_viz_hm_dry_spell_monthly_precip_mean_th{threshold}_adm1_southern_decjanfeb.png'))
# hm_monthly_southern_sel <-plot_heatmap_without_rainy(df_dry_spells,df_rainy_season_southern, dry_spell_match_values,match_values_labels,color_scale,y_label,plot_title,output_path_hm,ds_flatdata=TRUE,sub_title=sub_title)
# hm_monthly_southern_sel
# ggsave(output_path_hm,plot = hm_monthly_southern_sel, width=25,height=15)
# 
# ### dryspell=<=4mm/day ADM1 monthly precipitation and dry spells. Only southern and dec,jan, feb
# #Set general variables for heatmap
# threshold<- 180
# #classify match dry spell and below threshold precip during dry season as dry season (occurred once on 16-03-2020)
# dry_spell_match_values=c(3,0,2,1)
# match_values_labels=c(glue("Observed dry spell of <=4mm/day and <={threshold} monthly precipitation"), glue("No observed dry spell of <=4mm/day and >{threshold} monthly precipitation"),"Observed dry spell of <=4mm/day",glue('<={threshold} mm monthly precipitation'))
# color_scale=c('#0063B3','#b3e7ff','#F2645A',"#78D9D1")
# y_label=""
# df_dry_spells <- read.csv(paste0(dry_spell_dir,'seasonal/',glue('monthly_dryspellobs_4mm_adm1_th{threshold}_southern_decjanfeb.csv')))
# df_rainy_season <- read.csv(paste0(dry_spell_dir, "rainy_seasons_detail_2000_2020_mean_back.csv"))
# #only  the southern region
# df_rainy_season_southern <- df_rainy_season %>% mutate(region = substr(pcode, 3, 3)) %>%  filter(region==3)
# plot_title=glue("Overlap observed dry spells with <=4mm/day and <={threshold} mm monthly precipitation for the Southern region")
# sub_title="The year corresponds to the start of the rainy season"
# output_path_hm=paste0(exploration_dry_spell_dir, glue('mwi_viz_hm_dry_spell_monthly_precip_mean_4mm_th{threshold}_adm1_southern_decjanfeb.png'))
# 
# hm_monthly_southern_sel <-plot_heatmap_without_rainy(df_dry_spells,df_rainy_season_southern, dry_spell_match_values,match_values_labels,color_scale,y_label,plot_title,output_path_hm,ds_flatdata=TRUE,sub_title=sub_title)
# hm_monthly_southern_sel
# ggsave(output_path_hm,plot = hm_monthly_southern_sel, width=25,height=15)


### Only southern and Dec, Jan, Feb: Below threshold monthly observed and forecasted precipitation on ADM1
#Set general variables for heatmap
threshold<- 180
perc_th_value<- 45
lt <- 6

#classify match dry spell and below threshold precip during dry season as dry season (occurred once on 16-03-2020)
dry_spell_match_values=c(3,0,2,1)
match_values_labels=c(glue("<={threshold} observed and <={threshold} forecasted"), glue(">{threshold} observed and >{threshold} forecasted"),glue("<={threshold} observed"),glue('<={threshold} forecasted'))
color_scale=c('#0063B3','#b3e7ff','#F2645A',"#78D9D1")
y_label=""
df_dry_spells <- read.csv(paste0(exploration_dry_spell_dir,'monthly_precipitation/',glue('monthly_precip_obsfor_lt{lt}_th{threshold}_perc_{perc_th_value}_southern_decjanfeb.csv')))
df_rainy_season <- read.csv(paste0(dry_spell_dir, "rainy_seasons_detail_2000_2020_mean_back.csv"))
#only  the southern region
df_rainy_season_southern <- df_rainy_season %>% mutate(region = substr(pcode, 3, 3)) %>%  filter(region==3)

plot_title=glue("Overlap observed and forecasted <={threshold} mm monthly precipitation for the Southern region with leadtime = {lt} months")
sub_title="The year corresponds to the start of the rainy season"
output_path_hm=paste0(exploration_dry_spell_dir, 'monthly_precipitation/',glue('mwi_viz_hm_monthly_precip_obsfor_mean_th{threshold}_perc_{perc_th_value}_lt{lt}_adm1_southern_decjanfeb.png'))
hm_monthly_obsfor <-plot_heatmap_without_rainy(df_dry_spells,df_rainy_season_southern, dry_spell_match_values,match_values_labels,color_scale,y_label,plot_title,output_path_hm,ds_flatdata=TRUE,sub_title=sub_title)
hm_monthly_obsfor
ggsave(output_path_hm,plot = hm_monthly_obsfor, width=25,height=15)







# # Line plot
# # implemented for computing historical dry spells with mean aggregation on admin2
# df_rainy_season <- read.csv(paste0(dry_spell_dir, "rainy_seasons_detail_2000_2020_mean_back.csv"))
# df_dry_spells_mean <- read.csv(paste0(dry_spell_dir,'dry_spells_during_rainy_season_list_2000_2020_mean_back.csv'))
# df_ds <- fill_dates(df_dry_spells_mean, df_rainy_season, 'dry_spell_first_date', 'dry_spell_last_date', 1)
# df_ds %>%
#   mutate(region = as.factor(region))%>%
#   mutate(region = forcats::fct_relevel(region, 'Northern', 'Central', 'Southern'))%>%
#   group_by(dates, region)%>%
#   summarize(count_spells = sum(day_type))%>%
#   mutate(month_day = format(dates, "%m-%d"))%>%
#   mutate(date_no_year = as.Date(paste0('1800-',month_day), "%Y-%m-%d"))%>%
#   mutate(year = lubridate::year(dates))%>%
#   ggplot(aes(x=date_no_year, y=count_spells, group=year))+
#   geom_line(aes(color=year))+
#   facet_grid(rows=vars(region))+
#   scale_x_date(date_labels = "%b")+
#   theme_minimal()+
#   labs(title='Number of dry spells by region in Malawi, 2000-2020',
#        x='Date',
#        y='Count')
# 
# #ggsave(paste0(dry_spell_dir, '/dry_spell_plots/mean_back_dry_spell_line.png'))
# #ggsave(paste0(dry_spell_dir, '/dry_spell_plots/dry_spell_line_consec_days_4mm.png'), width = 7.55, height = 7.82, units = "in", dpi = 300)
# # ggsave(paste0(dry_spell_dir, '/dry_spell_plots/dry_spell_line_consec_days_2mm.png'), width = 7.55, height = 7.82, units = "in", dpi = 300)