#need to install the package patchwork
# install.packages("devtools")
# devtools::install_github("thomasp85/patchwork")

library(tidyverse)
library(sf)
library(glue)
library(patchwork)

# set path variables
data_dir <- Sys.getenv("AA_DATA_DIR")
shapefile_path <- paste0(data_dir, "/public/raw/eth/cod_ab/ET_Admin_OCHA_2020")

# read data
fn_path <-paste0(data_dir, "/public/processed/eth/fewsnet/cod_ps/ethiopia_fewsnet_admin1.csv")
gipc_path <- paste0(data_dir, "/public/processed/eth/ipc_global/ethiopia_globalipc_admin1.csv")
eth_adm1 <- st_read(paste0(shapefile_path, "/eth_admbnda_adm1_csa_bofed_20201008.shp"))

# set source variable
source <- "gipc" #fn"

if (source=="fn") {
  fn_all <- read.csv(fn_path)
  # compute percentages
  df_fi <- fn_all %>%
                mutate(perc_CS_3p = round((CS_3 + CS_4 + CS_5) * 100 / pop_CS, 1),
                       perc_CS_4p = round((CS_4 + CS_5) * 100 / pop_CS, 1),
                       perc_ML1_3p = round((ML1_3 + ML1_4 + ML1_5) * 100 / pop_ML1, 1),
                       perc_inc_ML1_3p = perc_ML1_3p - perc_CS_3p,
                       period_ML1 = "Oct - Jan 2021",
                       perc_ML2_3p = round((ML2_3 + ML2_4 + ML2_5) * 100 / pop_ML2, 1),
                       perc_inc_ML2_3p = perc_ML2_3p - perc_CS_3p,
                       period_ML2 = "Feb - May 2021",
                       perc_ML1_4p = round((ML1_4 + ML1_5) * 100 / pop_ML1, 1),
                       perc_inc_ML1_4p = perc_ML1_4p - perc_CS_4p,
                       perc_ML2_4p = round((ML2_4 + ML2_5) * 100 / pop_ML2, 1),
                       perc_inc_ML2_4p = perc_ML2_4p - perc_CS_4p)
} else if (source=="gipc"){
  df_fi <- read.csv(gipc_path)
  df_fi <- df_fi %>% 
    rename(
      ADM1_EN=ADMIN1
    )
  
} else{
  stop("Unknown source of data")
}
# subset vector data used to activate in December 2020
df_fisel <- df_fi %>%
  filter(date == '2020-10-01')
#determine projection periods, to be used for plot titles
period_ML1_str=unique(df_fisel$period_ML1)
period_ML2_str=unique(df_fisel$period_ML2)

summary(df_fisel) # note the absence of population in IPC4+        

####### 
### Generate plots
####### 

### IPC3+

#set colorscale limit
li = c(0, max(df_fisel %>% select(perc_CS_3p,perc_ML1_3p), na.rm = TRUE)+5)
# li = c(0,100)
br = c(0, 20, 40, 60, 80, 100)

# Current situation
fi_cs_3p_plot <- eth_adm1[, c('ADM1_EN', 'geometry')] %>% 
                 left_join(df_fisel, by = c('ADM1_EN' = 'ADM1_EN')) %>%
                 ggplot() +
                 geom_sf(aes(fill = perc_CS_3p), show.legend = F) +
                 scale_fill_gradient(low = "white", 
                                     high = "red", 
                                     na.value = NA, 
                                     "Percentage of regional population",
                                      breaks=br,
                                      limits=li) +
                 ggtitle('Observed by Oct 2020') + 
                         #subtitle = "Crisis or worse levels as reported by FewsNet") +
                 theme(axis.text.x=element_blank(),
                       axis.ticks.x=element_blank(),
                       axis.text.y=element_blank(),
                       axis.ticks.y=element_blank(),
                       panel.background = element_blank(),
                       panel.grid.major = element_blank(),
                       plot.title = element_text(size=28,margin=margin(30,0,0,0)))

# at ML1
#Have to manually set legend here to false if plotting two projection periods..
fi_ml1_3p_plot <- eth_adm1[, c('ADM1_EN', 'geometry')] %>% 
                left_join(df_fisel, by = c('ADM1_EN' = 'ADM1_EN')) %>%
                ggplot() +
                geom_sf(aes(fill = perc_ML1_3p), show.legend = T) +   
            #    geom_sf_label(aes(label = ADM1_EN), color = "black") +
                scale_fill_gradient(low = "white", 
                                    high = "red", 
                                    na.value = NA, 
                                    "Percentage of \n regional population",
                                    breaks=br,
                                    limits=li) +
                ggtitle(glue('Projected for {period_ML1_str}')) +
                        #, subtitle = "Crisis or worse levels as reported by FewsNet") +
                theme(axis.text.x=element_blank(),
                      axis.ticks.x=element_blank(),
                      axis.text.y=element_blank(),
                      axis.ticks.y=element_blank(),
                      panel.background = element_blank(),
                      panel.grid.major = element_blank(),
                      plot.title = element_text(size=28,margin=margin(30,0,0,0)),
                      legend.title = element_text(size=20),
                      legend.key.size = unit(2, 'cm'),
                      legend.text = element_text(size=20))

# increase by ML1
fi_inc_ml1_3p_plot <- eth_adm1[, c('ADM1_EN', 'geometry')] %>% 
                    left_join(df_fisel, by = c('ADM1_EN' = 'ADM1_EN')) %>%
                    ggplot() +
                    geom_sf(aes(fill = perc_inc_ML1_3p), show.legend = T) +            
                    scale_fill_gradient(low = "white", 
                                        high = "red", 
                                        na.value = NA, 
                                        "Percentage of regional population",
                                        breaks=br,
                                        limits=li) +
                    ggtitle('Projected Food Insecurity Increase - Oct-Jan 2020', subtitle = "Crisis or worse levels as reported by FewsNet") +
                    theme(axis.text.x=element_blank(),
                          axis.ticks.x=element_blank(),
                          axis.text.y=element_blank(),
                          axis.ticks.y=element_blank(),
                          panel.background = element_blank(),
                          panel.grid.major = element_blank())

# at ML2
fi_ml2_3p_plot <- eth_adm1[, c('ADM1_EN', 'geometry')] %>%
                  left_join(df_fisel, by = c('ADM1_EN' = 'ADM1_EN')) %>%
                  ggplot() +
                  geom_sf(aes(fill = perc_ML2_3p), show.legend = T) +
               #   geom_sf_label(aes(label = ADM1_EN), color = "black") +
                  scale_fill_gradient(low = "white",
                                      high = "red",
                                      na.value = NA,
                                      "Percentage of \n regional population",
                                      breaks=br,
                                      limits=li) +
                  ggtitle(glue('Projected for {period_ML2_str}')) +
                  #, subtitle = "Crisis or worse levels as reported by FewsNet") +
                  theme(axis.text.x=element_blank(),
                        axis.ticks.x=element_blank(),
                        axis.text.y=element_blank(),
                        axis.ticks.y=element_blank(),
                        panel.background = element_blank(),
                        panel.grid.major = element_blank(),
                        plot.title = element_text(size=28,margin=margin(30,0,0,0)),
                        legend.title = element_text(size=20),
                        legend.key.size = unit(2, 'cm'),
                        legend.text = element_text(size=20))

#########
#### create plots
#########

# dec 2020 food insecurity
png(file = glue("blog_post_food_insecurity_graph_{source}.png"),
   width=1820, height=750)
if (source =="fn"){
  layout_3p <- fi_cs_3p_plot | fi_ml1_3p_plot | fi_ml2_3p_plot
  caption = 'Data source: FewsNet'
} else if (source=="gipc"){
  layout_3p <- fi_cs_3p_plot | fi_ml1_3p_plot
  caption = 'Data source: Global IPC'
}

layout_3p +
  plot_annotation(
    title = 'Timeline of Food Insecurity in Ethiopia',
    subtitle = 'Crisis (IPC 3) or worse levels',
    caption = caption,
    theme = theme(plot.title = element_text(size = 40),
                  plot.subtitle = element_text(size = 24),
                  plot.caption = element_text(size=20))
  )
dev.off()


if (source=="fn"){
  # historical food insecurity reported in october report for 5 kililoch that activated in 2020 + 1 for a 3x2 graph
  hist_cs <- df_fi %>%
                  filter(date %in% c('2009-10-01', '2010-10-01', '2011-10-01', '2012-10-01', '2013-10-01', '2014-10-01', '2015-10-01', '2016-10-01', '2017-10-01', '2018-10-01', '2019-10-01', '2020-10-01'),
                        ADM1_EN %in% c('Afar', 'Oromia', 'Somali', 'Tigray', 'SNNP', 'Amhara')) %>%
                  select(date, ADM1_EN, pop_CS, CS_1, CS_2, CS_3, CS_4, CS_5, perc_CS_3p, perc_CS_4p) %>%
                  mutate(perc_CS_3 = round(100 * CS_3 / pop_CS, 1),
                         perc_CS_4 = round(100 * CS_4 / pop_CS, 1))

  hist_cs$date <- as.Date(hist_cs$date)

  hist_cs.long <- hist_cs[, c('date', "perc_CS_3", "perc_CS_4", "ADM1_EN")] %>%
                      rename('Crisis-IPC3' = 'perc_CS_3', 'Emergency-IPC4' = 'perc_CS_4') %>%
                      gather(key = "ipc_phase" , value = "perc", -date, -ADM1_EN)


  #png(file = "blog_post_historical_food_insecurity_graph.png",
  #    width=1051, height=578)

  ggplot(data = hist_cs.long[order(hist_cs.long$ipc_phase, decreasing = T),], # orders to plot IPC4 on top of IPC3
         aes(x = date, y = perc, fill = factor(ipc_phase, levels=c("Emergency-IPC4", "Crisis-IPC3")))) + # specifies order of the levels as we want them displayed
    geom_bar(stat="identity", na.rm = TRUE) +
    facet_wrap(~ ADM1_EN) +
    theme_bw() +
    ggtitle("Historical Food Insecurity in Select Regions",
            subtitle = "IPC3 levels in October as reported by FewsNet") +
    xlab("Year") +
    ylab("Regional population (%)") +
    ylim(0, 100) +
    scale_fill_manual(values = c("#BA0A1B", "#E69F00")) +
    scale_x_date(breaks = seq.Date(as.Date("2009-10-01"), as.Date("2020-10-01"), by = 'year'),
                 labels = lubridate::year(seq.Date(as.Date("2009-10-01"), as.Date("2020-10-01"), by = 'year'))) +
    theme(axis.text.x = element_text(angle = 60, hjust = 1)) +
    labs(fill = "IPC level")

  dev.off()
} else{ print(glue("Historical food insecurity not implemented for {source}"))}
## triggered areas (taken from POC dashboard)

ipc_indices_data <- read.csv("../../dashboard/data/foodinsecurity/ethiopia_foodinsec_trigger.csv")

# convert date string as a Date format
ipc_indices_data$date <- as.Date(ipc_indices_data$date, format = "%Y-%m-%d")

# select latest records. Last date of fewsnet and global ipc can differ so selected separately.
latest_report_per_source <- ipc_indices_data %>%
                                 group_by(source) %>%
                                 slice(which.max(date)) %>% # keep only latest records for each source
                                 ungroup() %>%
                                 select(source, date) %>%
                                 unique()

ipc_indices_data_latest <- ipc_indices_data %>%
                               right_join(latest_report_per_source, by = c('source' = 'source', 'date' = 'date'))
# build datasets per country, source. Ensures all regions are represented if no projections in certain regions
latest_fs <-  eth_adm1 %>%
                left_join(ipc_indices_data_latest, by = c('ADM1_EN' = 'ADMIN1'))


# generate list of triggered regions across sources. Feb-May 2021 is ML2 for FewsNet and Jan-June 2021 is ML1 for GlobalIPC
fs_trigger_list <- latest_fs %>%
  mutate(threshold_reached_H1_2021 = ifelse((source == 'FewsNet' & threshold_reached_ML2 == 'True') | (source == 'GlobalIPC' & threshold_reached_ML1 == 'True'), 1, 0)) %>%
  group_by(ADM1_EN) %>%
  mutate(threshold_reached_H1_2021 = ifelse(sum(threshold_reached_H1_2021) > 0, 1, 0)) %>% # assigns 1 to threshold_reached_H1_2021 if either source met threshold
  ungroup() %>%
  select(Shape_Leng, Shape_Area, ADM1_EN, threshold_reached_H1_2021, geometry) %>%
  unique()

# produce  map of triggered regions across sources
trigger_palette <- c("#EEEEEE", "#F2645A") # grey first, tomato second

fs_trigger_map <- fs_trigger_list %>%
  tmap::tm_shape() +
  tmap::tm_polygons("threshold_reached_H1_2021",
              palette = trigger_palette,
              title = "Food Insecurity Threshold Met",
              legend.show = FALSE) +
  tmap::tm_text(text = "ADM1_EN", size = 0.75, col = "black")


