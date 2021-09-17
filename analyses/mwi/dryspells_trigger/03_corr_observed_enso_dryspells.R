library(tidyverse)
library(ggplot2)
library(flextable)
library(glue)

# Set parameters
VERSION = 1

# Set data directories.
data_dir <- Sys.getenv("AA_DATA_DIR")
dry_spell_dir <- paste0(data_dir, glue('/public/processed/mwi/dry_spells/v{VERSION}/'))
enso_dir <- paste0(data_dir,'/public/raw/glb/enso/')
# filenames depend on aggregation methodology of dry spells from raster to admin/
plot_dir <-paste0(data_dir, glue('/public/processed/mwi/plots/dry_spells/v{VERSION}/enso/'))
dry_spells_list_path <- paste0(dry_spell_dir, 'archive/dry_spells_during_rainy_season_list_2000_2020_mean_back.csv')

############################# Definitions ##############################################
#3 month periods within rainy season were set as [SON, OND, NDJ, DJF, JFM, FMA, MAM].
#We often refer to these 3month periods as "season", not to be conused with the rainy season (indicated by season_approx)
#dry spells are considered as single records in the ground truth dataset(i.e they could be overlapping in year, season and duration)
#define the season_approx belonging to each entry, where the season_approx indicates the year during which the rainy season started
 ###DJF 2000 = Decem 1999, Jan Feb 2000
 ###NDJ 2000 = Nov Dec 2000, Jan 2001
########################################################################################
#month season mapping
mon_season <- data.frame(
  season = c ("DJF","JFM","FMA","MAM","AMJ","MJJ","JJA","JAS","ASO","SON","OND","NDJ"),
  month = c(1:12))

#for plotting
season_order <- c("SON", "OND", "NDJ", "DJF", "JFM", "FMA", "MAM")
enso_fill=c("#b52722", "#cccccc","#0063b3")
########################################################################################
############################# Preparing ENSO Data ######################################
########################################################################################
#getting enso data.
enso_oni <- read_csv(file=paste0(enso_dir,"ENSO_ONI_2000_2020.csv"))
enso_oni <- unite(enso_oni, enso_state, c(la_nina,neutral,el_nino))

#assigning labels of the corresponding enso states
enso_oni$enso_state[enso_oni$enso_state == '1_0_0'] <- 'La Nina'
enso_oni$enso_state[enso_oni$enso_state == '0_1_0'] <- 'Neutral'
enso_oni$enso_state[enso_oni$enso_state == '0_0_1'] <- 'El Nino'

#add a column with the middle-month of the season to the enso data
#this is later used to match dry spells
enso_oni <-merge(enso_oni, mon_season, by = c("season"), all.x = TRUE)

#get the dominant enso state per season_approx, where the dominant state is defined as the state occuring the majority of seasons
enso_oni$season_approx <-ifelse(enso_oni$month >= 10, enso_oni$year, ifelse(enso_oni$month <= 7, enso_oni$year - 1,'outside_rainy_season'))

#simplifying table for output
enso_oni<-enso_oni[,-which(names(enso_oni) == "month")]

#output table
ft <- flextable(enso_oni)
ft

#count occurrences of each enso state (one occurrence=year-season combination)
#this includes all data from 2000 till 2020, some of this data is not included in the dry spell list (i.e. first half of 2000 and second half 2020)
enso_years <- enso_oni %>%
  group_by(enso_state) %>%
  summarise(year_season = n())

ft <- flextable(enso_years)
ft


# Graphs
#plot the ONI value per year per season
# factor(m_enso_dryspell_sel$season, levels = season_order)
plt_oniseas <- ggplot(enso_oni, aes(fill=enso_state, y=anom, x=year)) +
    geom_bar(position="dodge", stat="identity") +
    scale_fill_manual(values=enso_fill)+
    facet_wrap(~factor(season, levels= season_order)) +
    theme(legend.position="none") +
    xlab("")
plt_oniseas

########################################################################################
############################# Preparing Dry Spells Data ################################
########################################################################################

#getting dry spells data.
dry_spells <- read_csv(dry_spells_list_path)

#extracting month from dry spell start date.
dry_spells$month <- format(as.Date(dry_spells$dry_spell_first_date, format="%Y-%m-%d"),"%m")
dry_spells$month <- sub("^0+", "", dry_spells$month)
#assign the dry spell to the season for which the month of the start date of the dry spell is the middle month of the season
#e.g. dry spell starting on 13-03-2005 is assigned to the FMA season.
dry_spells <- merge(mon_season, dry_spells, by = c("month"), all = TRUE)

#considering each record in the ground truth data as individual dry spell observation
dry_spells$observed <- 'yes'
dry_spells$number_dry_spell_observation <- 1

#simplifying table for output
dry_spells  <- dry_spells %>% filter(dry_spell_duration != 0)
dry_spells<-dry_spells[,-which(names(dry_spells) == "month")]

nrow(dry_spells)
#focusing only on the southern region
dry_spells_southern  <- dry_spells %>% filter(region == "Southern")
nrow(dry_spells_southern)
#get season_approx - season combinations with dry spells and count how many adm2's had a dry spell during that season
dry_spell_coverage <- dry_spells_southern %>%
  group_by(season,season_approx,observed) %>%
  #no_adm2 indicates the distinct number of adm2's with a dry spell during a season_approx - season combination.
  #The sum of this is 41, which means in 1 case there were two dry spells starting during the same season in the same adm2
  summarise(no_adm2 = n_distinct(ADM2_EN), avg_duration = mean(dry_spell_duration), num_dryspell_observation = sum(number_dry_spell_observation))


#output table
ft <- flextable(dry_spell_coverage)
ft

########################################################################################
###################### Preparing combined ENSO and Dry Spells Data #####################
########################################################################################
#Note this currently only uses dry spells in the Southern region!
# Can easily be adjusted above
# But reasoning is that theoretically most effect of ENSO is expected in the south, plus almost all dry spells occured in the Southern region
# Therefore removing the Northern and Central region, makes comparison easier

#merging Enso and dry spells data based on season_approx and season.
merged_enso_dryspell <-merge(enso_oni, dry_spell_coverage, by = c("season_approx","season"), all = TRUE)
merged_enso_dryspell$observed[is.na(merged_enso_dryspell$observed)] <- 'no'

#only select season_approx for which full CHIRPS data was analyzed, i.e. starting in 2000 till starting in 2019.
m_enso_dryspell_sel <- merged_enso_dryspell %>% filter(merged_enso_dryspell$season_approx %in% 2000:2019)

#if there is no dry spell in season_approx-season, set avg dry spell duration to 0
m_enso_dryspell_sel$avg_duration[is.na(m_enso_dryspell_sel$avg_duration)] <- 0

#order the occurrence of column values
#used later for creating plots
m_enso_dryspell_sel$season <- factor(m_enso_dryspell_sel$season, levels = season_order)
m_enso_dryspell_sel$observed <- factor(m_enso_dryspell_sel$observed, levels = c("yes","no"))
m_enso_dryspell_sel$enso_state <- factor(m_enso_dryspell_sel$enso_state, levels = c("El Nino","Neutral","La Nina"))

#compute the number of season_approx - season combinations per enso state (season=3months period)
enso_season_approx_season <- m_enso_dryspell_sel %>%
  group_by(enso_state) %>%
  summarise(no_season_approx_season_enso = n())

#output table
ft <- flextable(enso_season_approx_season)
ft

#only select the season_approx-seasons with a dry spell
m_enso_dryspell_sel_subset  <- m_enso_dryspell_sel %>% filter(!is.na(no_adm2))

#the number of unique season_approx-season combinations during which a dry spell occurred by enso state
dryspell_season_approx_season <- m_enso_dryspell_sel_subset %>%
  group_by(enso_state) %>%
  summarise(no_season_approx_season_dryspell = n())

#output table
ft <- flextable(dryspell_season_approx_season)
ft


#1: Occurence of dry spells per ENSO state
#aggregated by season (3 month period)
########################################################################################
ft <- flextable(m_enso_dryspell_sel)
ft

#select the seasons (3month periods) which observed a dry spell during at least one rainy season
#else it will give a cluttered since it will include many seasons that are not relevant for the current definition of a dry spell
enso_season_ds_occured <- filter(m_enso_dryspell_sel, season %in% unique(m_enso_dryspell_sel_subset$season))
#compute the confusion matrix on whether a dry spell was observed and an enso state was observed, grouped by 3month period
enso_season_dryspell <- enso_season_ds_occured %>%
  group_by(enso_state,season) %>%
  summarise(num_season_enso = n(), ds_enso = sum(observed=="yes"),no_ds_enso = sum(observed=="no"))
enso_season_dryspell$sum_ds_seas <- ave(enso_season_dryspell$ds_enso, enso_season_dryspell$season, FUN=sum)
enso_season_dryspell$sum_no_ds_seas <- ave(enso_season_dryspell$no_ds_enso, enso_season_dryspell$season, FUN=sum)
#fn
enso_season_dryspell$ds_no_enso <- enso_season_dryspell$sum_ds_seas-enso_season_dryspell$ds_enso
#tn
enso_season_dryspell$no_ds_no_enso <- enso_season_dryspell$sum_no_ds_seas-enso_season_dryspell$no_ds_enso

#convert the confusion matrix to percentages
enso_season_dryspell$perc_enso_ds <- enso_season_dryspell$ds_enso/enso_season_dryspell$num_season_enso*100
enso_season_dryspell$perc_enso_no_ds <- enso_season_dryspell$no_ds_enso/enso_season_dryspell$num_season_enso*100
enso_season_dryspell$perc_ds_per_enso <- enso_season_dryspell$ds_enso/enso_season_dryspell$sum_ds_seas*100

#output table
ft <- flextable(enso_season_dryspell)
ft

#plot the percentage of rainy seasons that had a dry spell per enso state and per season (3months)
ggplot(data = enso_season_dryspell, aes(y = perc_enso_ds, x =enso_state, fill=enso_state )) +
        geom_bar(width=0.5, stat = "identity")+
        theme_minimal()+
        #order by rainy season occurrence
        facet_wrap(~factor(season, levels=c("NDJ","DJF","JFM","FMA","MAM"))) +
        scale_fill_manual(values=enso_fill)+
        ylim(0,100)+
        labs(title = "Percentage of rainy seasons with a dry spell", subtitle="by ENSO state per 3month period",
             x = "ENSO State",
             y = "Percentage of rainy seasons (%)")

#plot the percentage of rainy seasons with NO dry spell per enso state and per season (3months)
ggplot(data = enso_season_dryspell, aes(y = perc_enso_no_ds, x =enso_state, fill=enso_state )) +
  geom_bar(width=0.5, stat = "identity")+
  theme_minimal()+
  #order by rainy season occurrence
  facet_wrap(~factor(season, levels=c("NDJ","DJF","JFM","FMA","MAM"))) +
  scale_fill_manual(values=enso_fill)+
  ylim(0,100)+
  labs(title = "Percentage of rainy seasons without a dry spell", subtitle="by ENSO state per 3 month period",
       x = "ENSO State",
       y = "Percentage of rainy seasons (%)")

#plot which percentage of dry spells occurred during each enso state
ggplot(data = enso_season_dryspell, aes(y = perc_ds_per_enso, x =enso_state, fill=enso_state )) +
  geom_bar(width=0.5, stat = "identity")+
  theme_minimal()+
  #order by rainy season occurrence
  facet_wrap(~factor(season, levels=c("NDJ","DJF","JFM","FMA","MAM"))) +
  scale_fill_manual(values=enso_fill)+
  ylim(0,100)+
  labs(title = "The ENSO state during rainy seasons with a dry spell", subtitle="per 3 month period",
       x = "ENSO State",
       y = "Percentage of rainy seasons (%)")

# 2: ratio of dry spells by rainy season for the dominant ENSO state
#same as 1 but aggregated to year instead of 3month period
########################################################################################
#only using the 3month periods (seasons) during which a dry spell ever occurred
#however, using the whole rainy season (SON - MAM) returns the same result
enso_year <- enso_season_ds_occured%>% group_by(season_approx) %>% summarize (enso_state =names(which.max(table(enso_state))),num_ds = sum(observed=="yes"))
flextable(enso_year)
enso_year_dryspell <- enso_year %>% group_by(enso_state) %>%
  #ds_enso=tp
  summarize(num_year_enso=n(),ds_enso=sum(num_ds>0)) %>%
  #no_ds_enso=fp
  mutate(no_ds_enso=num_year_enso-ds_enso)

#fn
enso_year_dryspell$ds_no_enso <- sum(enso_year_dryspell$ds_enso)-enso_year_dryspell$ds_enso
#tn
enso_year_dryspell$no_ds_no_enso <- sum(enso_year_dryspell$no_ds_enso)-enso_year_dryspell$no_ds_enso

enso_year_dryspell$perc_enso_ds <- enso_year_dryspell$ds_enso/enso_year_dryspell$num_year_enso*100
enso_year_dryspell$perc_enso_no_ds <- enso_year_dryspell$no_ds_enso/enso_year_dryspell$num_year_enso*100
enso_year_dryspell$perc_ds_per_enso <- enso_year_dryspell$ds_enso/sum(enso_year_dryspell$ds_enso)*100

flextable(enso_year_dryspell)

### Graphs
#plot the percentage of rainy seasons with a dry spell per enso state and per season (3months)
plt_enso_ds <-ggplot(data = enso_year_dryspell, aes(y = perc_enso_ds, x =enso_state, fill=enso_state )) +
  geom_bar(width=0.5, stat = "identity", show.legend = F )+
  theme_minimal()+
  scale_fill_manual(values=enso_fill)+
  ylim(0,100)+
  labs(title = "Percentage of rainy seasons with a dry spell per ENSO state",
       x = "ENSO State",
       y = "Percentage of rainy seasons (%)")
plt_enso_ds
# ggsave(paste0(plot_dir, 'mwi_percentage_enso_state_dryspell.png'),plot=plt_enso_ds)

#plot the percentage of rainy seasons with NO dry spell per enso state and per season (3months)
plt_enso_no_ds <- ggplot(data = enso_year_dryspell, aes(y = perc_enso_no_ds, x =enso_state, fill=enso_state )) +
  geom_bar(width=0.5, stat = "identity")+
  theme_minimal()+
  scale_fill_manual(values=enso_fill)+
  ylim(0,100)+
  labs(title = "Percentage of rainy seasons without a dry spell",
       x = "ENSO State",
       y = "Percentage of rainy seasons (%)")

#plot which percentage of dry spells occured in each enso state
plt_ds_enso <- ggplot(data = enso_year_dryspell, aes(y = perc_ds_per_enso, x =enso_state, fill=enso_state)) +
  geom_bar(width=0.5, stat = "identity", show.legend = F )+
  theme_minimal()+
  scale_fill_manual(values=enso_fill)+
  ylim(0,100)+
  labs(title = "Percentage of dry spells occuring during each ENSO state",
       # caption="The ENSO state is the state that had the most occurences during the rainy season",
       x = "ENSO State",
       y = "Percentage of rainy seasons (%)")

#ggsave(paste0(plot_dir, 'mwi_percentage_dryspell_enso_state.png'),plot=plt_ds_enso)

#3: ONI values and dry spells
########################################################################################

#set values for horizontal intersection lines
line.data <- data.frame(yintercept = c(-0.5, 0.5), ENSO_state = c("La Niña", "El Niño"))  
plt_anom <- ggplot(m_enso_dryspell_sel, aes(x=season, y=anom,fill=observed)) +
  geom_hline(aes(yintercept = yintercept, color = ENSO_state), line.data)+
  scale_color_manual(values=c(first(enso_fill), last(enso_fill)))+
  geom_bar(width=0.5, stat = "identity")+
  facet_wrap(~season_approx)+
  labs(y='ONI', x='3-month period')+
  theme_bw()+
  labs(title="ONI values per rainy season", subtitle="The year indicates the start of the rainy season",fill="Dry spell observed")+
  scale_fill_manual(values=c("#F2645A", "#cccccc"))+
  theme(legend.position = 'bottom',
       axis.text.x = element_text(angle = 90,margin=margin(5,0,0,0)))
plt_anom
# ggsave(paste0(plot_dir, 'mwi_plot_oni_year_dryspell.png'))

#investigate if more dry spells occurred during more extreme ONI values
#this is not the case
m_enso_dryspell_sel_zero <- enso_season_ds_occured
m_enso_dryspell_sel_zero[is.na(m_enso_dryspell_sel_zero)] <-0
flextable(m_enso_dryspell_sel_zero)

plt_anomadm <- ggplot(m_enso_dryspell_sel_zero, aes(x=factor(no_adm2), y=anom)) +
  geom_violin()+
  labs(title="Distribution of ONI values grouped by the number of dry spells",x="Number of admin2's with a dry spell",y="ONI value")
plt_anomadm

#4: Geographical spread of dry spells (number of admin2s) per ENSO state
########################################################################################
plt_ensoadm <- ggplot(m_enso_dryspell_sel_zero, aes(x=factor(enso_state), y=no_adm2)) +
  geom_violin()+
  labs(title="Distribution of number of simeltaneous dry spells per ENSO state",x="ENSO state",y="Number of admin2's")
plt_ensoadm

