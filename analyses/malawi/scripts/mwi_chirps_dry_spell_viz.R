library(sf)
library(ggplot2)
library(tmap)
library(dplyr)
library(RcppRoll)
library(tidyr)

theme_set(theme_bw())

# Reading in the data -----------------------------------------------------

data_dir <- Sys.getenv("AA_DATA_DIR")
dry_spell_dir <- paste0(data_dir, '/processed/malawi/dry_spells/')
df_dry_spells_max <- read.csv(paste0(dry_spell_dir, 'dry_spells_during_rainy_season_list_2000_2020.csv'))
df_dry_spells_mean <- read.csv(paste0(dry_spell_dir, 'dry_spells_during_rainy_season_list_2000_2020_mean.csv'))
df_rainy_season_mean <- read.csv(paste0(dry_spell_dir, 'rainy_seasons_detail_2000_2020_mean.csv'))   
df_rainy_season_max <- read.csv(paste0(dry_spell_dir, 'rainy_seasons_detail_2000_2020.csv'))   
data_mean_long <- readRDS(paste0(data_dir, "/processed/malawi/dry_spells/data_mean_values_long.RDS"))# Fill in rainy or dry spell dates 


# Transform the data for plotting -----------------------------------------


fill_dates <- function(df_dates, start_col, end_col, fill_num){
  
  adms <- unique(data_mean_long[['pcode']])
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


df_ds <- fill_dates(df_dry_spells_max, 'dry_spell_first_date', 'dry_spell_last_date', 1)
df_rs <- fill_dates(df_rainy_season_max, 'onset_date', 'cessation_date', 10)


# Make the line plot and heatmap ------------------------------------------

# Heatmap
df_ds %>%
  full_join(df_rs, by=c('pcode', 'dates'))%>%
  mutate(days = as.factor(day_type.x + day_type.y))%>%
  mutate(days = factor(days, levels=c(10, 0, 11, 1), labels=c("Rainy season", "Dry season", 'Dry spell in rainy season', "Dry spell in dry season")))%>%
  ggplot(aes(x=date_no_year.x, y=pcode, fill=days))+
  geom_tile() +
  scale_fill_manual(values=c('#b3e7ff', '#fff2d6',"#b52722",  '#fc8d5d'))+
  facet_grid(rows=vars(year.x))+
  theme_minimal()+
  labs(title='20 years of dry spells in Malawi\'s Admin 2 regions, 2000-2020', x='Date', y='Admin 2 region', fill='')+
  theme(axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        legend.position = 'bottom')+
  scale_x_date(date_labels = "%b")
ggsave(paste0(dry_spell_dir, '/dry_spell_plots/max_dry_spell_hm.png'))

# Line plot
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
ggsave(paste0(dry_spell_dir, '/dry_spell_plots/max_dry_spell_line.png'))



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
