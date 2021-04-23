library(dplyr) 
library(tidyverse)
library(ggplot2)
library(flextable)
#library(metR)
library(lubridate)
library(zoo)
library(plotly)
library(scales)
library(viridis)
library(hrbrthemes)

# Set current working directory.
setwd("2021/pa-anticipatory-action/R_Analysis")
print(getwd())

############################# Definitions ##############################################
#mlw rainly 3months seasons [SON, OND, NDJ, DJF, JFM, FMA, MAM]
########################################################################################
#month season mapping
mon_season <- data.frame(
  season = c ("DJF","JFM","FMA","MAM","AMJ","MJJ","JJA","JAS","ASO","SON","OND","NDJ"), 
  month = c(1:12))
########################################################################################
############################# Preparing ENSO Data ######################################
########################################################################################
#getting enso data.
enso_oni <- read_csv("data/ENSO_ONI_2000_2020.csv")

enso_oni <- unite(enso_oni, enso_state, c(la_nina,neutral,el_nino))

#assigning labels of the corresponding enso states
enso_oni$enso_state[enso_oni$enso_state == '1_0_0'] <- 'La Nina'
enso_oni$enso_state[enso_oni$enso_state == '0_1_0'] <- 'Neutral'
enso_oni$enso_state[enso_oni$enso_state == '0_0_1'] <- 'El Nino'

#add a column with the middle-month of the season to the enso data
enso_oni <-merge(enso_oni, mon_season, by = c("season"), all.x = TRUE)

#get the dominant enso state per season_approx, where the dominant state is defined as the state occuring the majority of seasons
enso_oni$year <-ifelse(enso_oni$month >= 10, enso_oni$year, ifelse(enso_oni$month <= 7, enso_oni$year - 1,'outside_rainy_season'))

#simplifying table for output
enso_oni<-enso_oni[,-which(names(enso_oni) == "month")]

#output table
ft <- flextable(enso_oni)
ft

#count year season combinations per enso state
enso_years <- enso_oni %>%
  group_by(enso_state) %>%
  summarise(year_season = n())

ft <- flextable(enso_years)
ft

# bp<- ggplot(enso_years, aes(x="", y=years, fill=enso_state))+
#   geom_bar(width = 1, stat = "identity")
# bp
# pie <- bp + coord_polar("y", start=0)
# pie
# # Use custom color palettes
# pie + scale_fill_manual(values=c("#999999", "#E69F00", "#56B4E9"))

# Graphs

ggplot(data = enso_years, aes(x = enso_state, y =year_season, fill=enso_state )) +
  #geom_line(size = 2) +
  geom_bar(width=0.5, stat = "identity")+
  theme_minimal()+
  scale_fill_manual(values=c("#E69F00", "#56B4E9", "#999999"))+
  labs(title = "",
       x = "Enso State",
       y = "Number of year_season commbiations")


ggplot(data = enso_oni, aes(x = season, y =anom, fill=enso_state )) +
  #geom_line(size = 2) +
  geom_bar(width=0.5, stat = "identity")+
  theme_minimal()+
  scale_fill_manual(values=c("#E69F00", "#56B4E9", "#999999"))+
  labs(title = "",
       x = "Season",
       y = "ONI")


ggplot(enso_oni, aes(fill=enso_state, y=anom, x=year)) + 
    geom_bar(position="dodge", stat="identity") +
    #scale_fill_viridis(discrete = T, option = "E") +
    #ggtitle("Studying seasons..") +
    scale_fill_manual(values=c("#E69F00", "#56B4E9", "#999999"))+
    facet_wrap(~season) +
    theme_ipsum() +
    theme(legend.position="none") +
    xlab("")

########################################################################################
############################# Preparing Dry Spells Data ################################
########################################################################################

#getting dry spells data.
#dry_spells <- read.csv("data/dry_spells_during_rainy_season_list_2000_2020.csv") #old 
dry_spells <- read_csv("data/dry_spells_during_rainy_season_list_2000_2020_mean_back.csv") #new
#dry_spells <- read.csv("data/daily_mean_dry_spells_details_2mm_2000_2020.csv") #new 2mm
#dry_spells <- read.csv("data/daily_mean_dry_spells_details_2000_2020.csv") #new 4mm

#extracting month, season from dry spell start date.
dry_spells$month <- format(as.Date(dry_spells$dry_spell_first_date, format="%Y-%m-%d"),"%m")
dry_spells$month <- sub("^0+", "", dry_spells$month)  
dry_spells <- merge(mon_season, dry_spells, by = c("month"), all = TRUE)

dry_spells$observed <- 1

#simplifying table for output
dry_spells  <- dry_spells %>% filter(dry_spell_duration != 0)
dry_spells<-dry_spells[,-which(names(dry_spells) == "month")]

names(dry_spells)[names(dry_spells) == "season_approx"] <- "year"

dry_spells  <- dry_spells %>% filter(region == "Southern")

dry_spell_entry_ <- dry_spells %>%
  group_by(season,year) %>%
  summarise(no_adm2 = n_distinct(ADM2_EN), avg_duration = mean(dry_spell_duration))

dry_spell_entry$dry_spell_observed <- 'yes'
dry_spell_entry$number_dry_spell_observation <- 1

#merging Enso and dry spels data based on year and season.
m_enso_dryspell <-merge(enso_oni, dry_spell_entry, by = c("year","season"), all = TRUE)
m_enso_dryspell$dry_spell_observed[is.na(m_enso_dryspell$dry_spell_observed)] <- 'no'


m_enso_dryspell_sel <- m_enso_dryspell[!m_enso_dryspell$year %in% c(1999,2020),]

#output table
ft <- flextable(m_enso_dryspell)
ft

ggplot(data = m_enso_dryspell_sel, aes(x = dry_spell_observed, y = year ,fill=dry_spell_observed)) +
  geom_point(size=2, shape=23)+
  #geom_line(size = 2) +
 # geom_col(position = "dodge")+
  #geom_bar(width=0.5, stat = "identity")+
  theme_minimal()+
  labs(title = "",
       y = "year",
       x = "dry spell observed")

ggplot(data = m_enso_dryspell_sel, aes(x = dry_spell_observed, y = year ,fill=dry_spell_observed)) +
  geom_point(size=2, shape=23)+
  #geom_line(size = 2) +
  # geom_col(position = "dodge")+
  #geom_bar(width=0.5, stat = "identity")+
  theme_minimal()+
  labs(title = "",
       y = "year",
       x = "dry spell observed")

ggplot(data = m_enso_dryspell, aes(x = dry_spell_observed, y = season ,fill=dry_spell_observed)) +
  geom_point(size=2, shape=23)+
  #geom_line(size = 2) +
  # geom_col(position = "dodge")+
  #geom_bar(width=0.5, stat = "identity")+
  theme_minimal()+
  labs(title = "",
       y = "season",
       x = "dry spell observed")

dry_spells_season <- dry_spells %>%
  group_by(year, season,region,dry_spell_duration ) %>%
  summarise(number_of_dryspells = n_distinct(dry_spell_first_date,dry_spell_last_date))

dry_spells_region <- dry_spells %>%
  group_by(season, region) %>%
  summarise(number_of_dryspells = n_distinct(dry_spell_first_date,dry_spell_last_date))


dry_spells_duration <- dry_spells_season %>%
  group_by(year,region) %>%
  summarise(avg_dryspell_duration = mean(dry_spell_duration))

#output table
ft <- flextable(dry_spells_region)
ft

ggplot(data = dry_spells_region, aes(x = number_of_dryspells, y = season, fill=region)) +
  #geom_line(size = 2) +
  geom_col(width=0.5, position = "dodge")+
  theme_minimal()+
  labs(title = "",
       y = "Season",
       x = "Number of dry spells")

ggplot(data = dry_spells_region, aes(x = year, y = number_of_dryspells , fill=region)) +
  #geom_line(size = 2) +
  #geom_bar(width=0.5, stat = "identity")+
  geom_col(width=0.5, position = "dodge")+
  theme_minimal()+
  labs(title = "",
       x = "Year",
       y = "Number of dry spells")


########################################################################################
###################### Preparing combined ENSO and Dry Spells Data #####################
########################################################################################

#merging Enso and dry spels data based on year and season.
merged_enso_dryspell <-merge(enso_oni, dry_spells, by = c("year","season"), all = TRUE)

#if there is no dry spell, set dry spell duration to 0 
merged_enso_dryspell$dry_spell_duration[is.na(merged_enso_dryspell$dry_spell_duration)] <- 0

#filtering
merged_enso_dryspell  <- merged_enso_dryspell %>% filter(!is.na(region))

#output table
ft <- flextable(enso_state_dryspell__)
ft


enso_dryspell <- merged_enso_dryspell %>%
  group_by(year) %>%
  summarise(number_of_dryspells = n())


enso_dryspell_sel_year <- m_enso_dryspell_sel %>%
  group_by(year, enso_state) %>%
  summarise(no_dry_spell_observation = sum(number_dry_spell_observation,na.rm = TRUE), avg_oni = mean(anom))

enso_dryspell_sel <- m_enso_dryspell_sel %>%
  group_by(enso_state) %>%
  summarise(no_year_season_dry_spell =sum(number_dry_spell_observation,na.rm = TRUE))

enso_dryspell_sel_ <- m_enso_dryspell_sel %>%
  group_by(enso_state) %>%
  summarise(no_year_season_enso = n())

enso_dryspell_sel__ <- m_enso_dryspell_sel %>%
  group_by(enso_state) %>%
  summarise(no_adm2 = sum(no_adm2,na.rm = TRUE))

enso_dryspell_oni <- m_enso_dryspell_sel %>%
  group_by(enso_state,dry_spell_observed) %>%
  summarise(no_onis = n(), avg_oni = mean(anom), avg_duration = mean(avg_duration))

enso_state_dryspell_ <-merge(enso_dryspell_sel_, enso_dryspell_sel, by = c("enso_state"), all = TRUE)
enso_state_dryspell__ <-merge(enso_dryspell_sel__, enso_dryspell_sel_, by = c("enso_state"), all = TRUE)
enso_state_dryspell__$pct <- enso_state_dryspell__$no_adm2/enso_state_dryspell_$no_year_season_enso

enso_state_dryspell_$pctall <- enso_state_dryspell_$no_year_season_dry_spell*100/enso_state_dryspell_$no_year_season_enso
#filtering
enso_dryspell_sel  <- enso_dryspell_sel %>% filter(!is.na(no_adm2))

ggplot(data = enso_dryspell_sel_year, aes(x = year, y = no_dry_spell_observation, fill=enso_state)) +
  #geom_line(size = 2) +
  #geom_bar(width=0.5, stat = "identity")+
  geom_col(width=0.5, position = "dodge")+
  scale_fill_manual(values=c("#E69F00", "#56B4E9", "#999999"))+
  theme_minimal()+
  ylim(0,1)+
  labs(title = "",
       x = "year",
       y = "Number of dry spell observations")

ggplot(data = enso_state_dryspell_, aes(x = enso_state, y = pct, fill=enso_state)) +
  #geom_line(size = 2) +
  #geom_bar(width=0.5, stat = "identity")+
  geom_col(width=0.5, position = "dodge")+
  scale_fill_manual(values=c("#E69F00", "#56B4E9", "#999999"))+
  theme_minimal()+
  ylim(0,1)+
  labs(title = "",
       x = "Enso state",
       y = "Number of dry spell observations")




ggplot(data = enso_state_dryspell__, aes(y = pct, x =enso_state, fill=enso_state )) +
  #geom_line(size = 2) +
  geom_bar(width=0.5, stat = "identity")+
  theme_minimal()+
  scale_fill_manual(values=c("#E69F00", "#56B4E9", "#999999"))+
  ylim(0,1)+
  labs(title = "",
       x = "Enso State",
       y = "Pct(no_adm2/no_year_season_enso)")

enso_year_dryspell_group<- m_enso_dryspell_sel %>% group_by(enso_state) %>% summarize (fp=sum(is.na(number_dry_spell_observation)),tp=sum(!is.na(number_dry_spell_observation)))
enso_year_dryspell_group$fn<-(sum(enso_year_dryspell_group$tp)-enso_year_dryspell_group$tp)
enso_year_dryspell_group$tn<-(sum(enso_year_dryspell_group$fp)-enso_year_dryspell_group$fp)


ggplot(data = enso_dryspell, aes(x = year, y = number_of_dryspells)) +
  #geom_line(size = 2) +
  #geom_bar(width=0.5, stat = "identity")+
  geom_col(width=0.5, position = "dodge")+
  theme_minimal()+
  labs(title = "",
       x = "Year",
       y = "Number of dry spells")

ggplot(data = enso_dryspell_oni, aes(x = enso_state, y = avg_duration, fill=dry_spell_observed)) +
  #geom_line(size = 2) +
  #geom_bar(width=0.5, stat = "identity")+
  geom_col(width=0.5, position = "dodge")+
 # geom_point(size=2, shape=23)+
  theme_minimal()+
  labs(title = "",
       x = "Enso state",
       y = "avg_duration")

#mean on dry spell season year
dry_spells_enso_state <- merged_enso_dryspell %>%
  group_by(anom) %>%
  summarise(num_dryspells = n_distinct(dry_spell_first_date,dry_spell_last_date))

#mean on dry spell season year
dry_spells_enso_state <- merged_enso_dryspell %>%
  group_by(anom) %>%
  summarise(avg_dryspell_duration = mean(dry_spell_duration))

dry_spells_enso_state$observed <- ifelse(dry_spells_enso_state$num_dryspells==1,'Yes-1time',ifelse(dry_spells_enso_state$num_dryspells==2,'Yes-2times',ifelse(dry_spells_enso_state$num_dryspells==3,'Yes-3times',ifelse(dry_spells_enso_state$num_dryspells==4,'Yes-4times','No'))))

#summarizing on mlw rainly season
mlw_enso_dryspell <- merged_enso_dryspell %>%
  group_by(year, region, enso_state) %>%
  summarise(unique_seasons=n_distinct(season),  unique_admin2s= n_distinct(ADM2_EN, na.rm = TRUE),num_dryspells = sum(dry_spell_duration!=0))

#mean on dry spell season year
avg_dryspell <- mlw_enso_dryspell %>%
  group_by(enso_state, region) %>%
  summarise( avg_dryspells = mean(num_dryspells))

#print as table
ft <- flextable(enso_dryspell_oni)
ft

ft <- flextable(dry_spells_enso_state)
ft


enso_state_dryspell <-merge(dry_spells_enso_state, enso_years, by = c("enso_state"), all = TRUE)
enso_state_dryspell$pct <- enso_state_dryspell$number_of_dryspells/enso_state_dryspell$year_season

ggplot(data = enso_state_dryspell, aes(x = pct, y =enso_state, fill = region )) +
  #geom_line(size = 2) +
  geom_bar(width=0.5, stat = "identity")+
  theme_minimal()+
  #scale_fill_manual(values=c("#E69F00", "#56B4E9", "#999999"))+
  labs(title = "",
       x = "Enso State",
       y = "Pct(dry_spell_enso/enso_year_season)")


ggplot(data = mlw_enso_dryspell, aes(x = enso_state, y =num_dryspells, fill = region )) +
  #geom_line(size = 2) +
  geom_bar(width=0.5, stat = "identity")+
  theme_minimal()+
  scale_fill_brewer(palette="Reds")+
  labs(title = "",
       x = "Enso State",
       y = "Number of dry spells")

ggplot(data = dry_spells_enso_state, aes(x = enso_state, y =dry_spell_duration, fill=observed)) +
  #geom_line(size = 2) +
  #geom_col(aes(y=observed)) +
  geom_col(width=0.5, position = "dodge")+
  
  theme_minimal()+
  #scale_fill_manual(values=c("#E69F00", "#56B4E9", "#999999"))+
  scale_fill_brewer(palette="Reds")+
  labs(title = "",
      x = "ENSO state",
      y = "Dry spell duration")

ggplot(data = dry_spells_enso_state, aes(x = anom, y =season, fill=observed)) +
  #geom_line(size = 2) +
  #geom_col(aes(y=observed)) +
  geom_col(width=0.5, position = "dodge")+
  
  theme_minimal()+
  #scale_fill_manual(values=c("#E69F00", "#56B4E9", "#999999"))+
 scale_fill_brewer(palette="Reds")+
  labs(title = "",
       y = "Season",
       x = "ONI")

boxplot(anom ~ num_dryspells, data = dry_spells_enso_state,
        varwidth = TRUE, log = "x", las = 1)


########################################################################################
#Getting seasons where dry spells are frequent against all in enso and mlw rainy season #
########################################################################################

#summarizing on dry spell seasons
count_seasons <- merged_enso_dryspell %>%
  group_by(enso_state, region) %>%
  summarise(unique_season_year_combination =n_distinct(season,year))


merged_seasons_dryspell <-merge(count_seasons, count_enso_States, by = c("enso_state"), all = TRUE)

merged_seasons_dryspell$ratio <- merged_seasons_dryspell$unique_season_year_combination/merged_seasons_dryspell$total_season_year_combinations

ft <- flextable(merged_seasons_dryspell)
ft

ggplot(data = merged_seasons_dryspell, aes(x = enso_state, y =ratio, fill = region )) +
  #geom_line(size = 2) +
  geom_col(width=0.5, position = "dodge")+
  theme_minimal()+
  scale_fill_brewer(palette="Reds")+
  labs(title = "",
       x = "Enso State",
       y = "Ratio(season year in dryspells/total")

########################################################################################
################ Getting frequent seasons in dry spells over total #####################
########################################################################################

myvars <- c("season", "enso_state", "region","dry_spell_duration")
seasons <-merged_enso_dryspell[myvars]

#summarizing on dry spell seasons
count_seasons_ <- seasons %>%
  group_by(season, enso_state) %>%
  summarise('dryspell_season'=n())

year_season_state<- merged_enso_dryspell %>%
  group_by(season, year, enso_state) %>%
  summarise('dryspell_season'=n())

count_seasons_<-year_season_state%>%
  group_by(season, enso_state) %>%
  summarise('dryspell_season'=n())


merged_seasons <-merge(count_seasons_,count_season, by = c("enso_state","season"), all = TRUE)

#filtering
merged_seasons  <- merged_seasons %>% filter(!is.na(dryspell_season))

merged_seasons$ratio <- merged_seasons$dryspell_season/merged_seasons$`20yrs_total_season`

ft <- flextable(merged_seasons)
ft

ggplot(data = merged_seasons, aes(x = enso_state, y = ratio, fill = season )) +
  #geom_line(size = 2) +
  geom_col(width=0.5, position = "dodge")+
  theme_minimal()+
 # scale_fill_discrete(labels = c("NDJ" , "DJF" , "JFM" , "FMA" ))+
  #scale_fill_manual("Legenda", values = c("Outlier" = "#1260AB", "NOutlier" = "#009BFF"))
  labs(title = "",
       x = "Enso State",
       y = "Ratio(enso seasons dryspell/total")

########################################################################################

#add a column with the middle-month of the season to the enso data
enso_oni_month <-merge(enso_oni, mon_season, by = c("season"), all.x = TRUE)
#define the season_approx belonging to each entry, where the season_approx indicates the year during which the rainy season started
#DJF 2000 = Decem 1999, Jan Feb 2000
#NDJ 2000 = Nov Dec 2000, Jan 2001

enso_oni_month$season_approx <-ifelse(enso_oni_month$month >= 10, enso_oni_month$year, ifelse(enso_oni_month$month <= 7, enso_oni_month$year - 1,'outside_rainy_season'))
#get the dominant enso state per season_approx, where the dominant state is defined as the state occuring the majority of seasons

enso_year <- enso_oni_month%>% group_by(season_approx) %>% summarize (enso_state =names(which.max(table(enso_state))))
#compute whether a dry spell occured or not during a season_approx
#only focussing on the southern region, since this has according to literature the most impact of ENSO, and almost all dry spells occur in that region

dry_spells_year <- dry_spells %>% filter(region=="Southern") %>% group_by(season_approx) %>% summarize(dry_spell=sum(dry_spell_duration,na.rm = TRUE))
#merge enso and dry spells

enso_dryspell_year <- merge(enso_year, dry_spells_year, by = c("season_approx"), all = TRUE)
#dont have dry spell data for 1999 and 2020, so remove them from the enso data (to prevent false false alarms)

enso_dryspell_year_sel <- enso_dryspell_year[!enso_dryspell_year$season_approx %in% c(1999,2020),]
#compute the fp,tp,fn,tn per enso state over the years

enso_year_dryspell_group<- enso_dryspell_year_sel %>% group_by(enso_state) %>% summarize (fp=sum(is.na(dry_spell)),tp=sum(!is.na(dry_spell)))
enso_year_dryspell_group$fn<-(sum(enso_year_dryspell_group$tp)-enso_year_dryspell_group$tp)
enso_year_dryspell_group$tn<-(sum(enso_year_dryspell_group$fp)-enso_year_dryspell_group$fp)

# write.csv(enso_year_dryspell_group, file=paste0(dry_spell_dir,"enso/enso_dominant_state_confusion.csv"))
#compute the false alarm rate and hit rate and plot them
enso_year_dryspell_group$falsealarm_rate <- enso_year_dryspell_group$tp/(enso_year_dryspell_group$tp+enso_year_dryspell_group$fp                                                )
enso_year_dryspell_group$hit_rate <- (enso_year_dryspell_group$tp/(enso_year_dryspell_group$tp+enso_year_dryspell_group$fn))

ft <- flextable(enso_year_dryspell_group)
ft

ggplot(data = enso_year_dryspell_group, aes(x = enso_state, y = hit_rate, fill=enso_state )) +
  #geom_line(size = 2) +
  geom_col(width=0.5, position = "dodge")+
  theme_minimal()+
  ylim(0,1)+
  scale_fill_manual(values=c("#E69F00", "#56B4E9", "#999999"))+
  labs(title = "",
       x = "Enso State",
       y = "Probability of detecting a dry spell")

ggplot(data = enso_year_dryspell_group, aes(x = enso_state, y = falsealarm_rate,  fill=enso_state )) +
  #geom_line(size = 2) +
  geom_col(width=0.5, position = "dodge")+
  theme_minimal()+
  ylim(0,1)+
  scale_fill_manual(values=c("#E69F00", "#56B4E9", "#999999"))+
  labs(title = "",
       x = "Enso State",
       y = "Probability of giving a false alarm")

########################################################################################

#barplot(count_enso_States,"enso states","number of events", col = enso_state)
ggplot(data = mlw_enso_dryspell, aes(x = enso_state, y = num_dryspells, fill = region )) +
  #geom_line(size = 2) +
  geom_col(width=0.5, position = "dodge")+
  theme_minimal()+
  scale_fill_brewer(palette="Reds")+
  geom_text(aes(label=num_dryspells))+
  labs(title = "",
       x = "Enso State",
       y = "Num_dryspells")

ggplot(data = count_seasons, aes(x = enso_state, y = unique_season_year_combination, fill = region )) +
  #geom_line(size = 2) +
  geom_col(width=0.5, position = "dodge")+
  theme_minimal()+
  scale_fill_brewer(palette="Reds")+
  labs(title = "",
       x = "Enso State",
       y = "Unique Seasons")

write.csv(merged_enso_dryspell,"./data/mar28/mlw_enso_dryspells_back_mar29_season_approx.csv", row.names = TRUE)
write.csv(enso_oni_month,"./data/mar28/mlw_enso_oni_mar29_season_approx.csv", row.names = TRUE)

p <- ggplot(merged_enso_dryspell, aes(class, hwy))
p + geom_boxplot()
