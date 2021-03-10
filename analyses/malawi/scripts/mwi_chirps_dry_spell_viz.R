library(sf)
library(ggplot2)
library(tmap)

theme_set(theme_bw())

# Reading in the data -----------------------------------------------------

data_dir <- Sys.getenv("AA_DATA_DIR")
dry_spell_dir <- paste0(data_dir, '/processed/malawi/dry_spells/')
df_dry_spells_max <- read.csv(paste0(dry_spell_dir, 'dry_spells_during_rainy_season_list_2000_2020.csv'))
df_dry_spells_mean <- read.csv(paste0(dry_spell_dir, 'dry_spells_during_rainy_season_list_2000_2020_mean.csv'))
df_rainy_season <- read.csv(paste0(dry_spell_dir, 'rainy_seasons_detail_2000_2020_mean.csv'))                            


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
 
data_mean_long <- readRDS(paste0(data_dir, "/processed/malawi/dry_spells/data_mean_values_long.RDS"))

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

# Making some maps --------------------------------------------------------

# How many total dry spells per admin 2 region?


