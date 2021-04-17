library(tidyverse)
library(sf)

# set path variables
data_dir <- Sys.getenv("AA_DATA_DIR")
shapefile_path <- paste0(data_dir, "/raw/ethiopia/Shapefiles/ET_Admin_OCHA_2020")

# read data
fn_all <- read.csv(paste0(data_dir, "/processed/ethiopia/FewsNetAdmPop/ethiopia_fewsnet_admin1.csv"))

eth_adm1 <- st_read(paste0(shapefile_path, "/eth_admbnda_adm1_csa_bofed_20201008.shp"))

# subset vector data used to activate in December 2020
fn <- fn_all %>%
        filter(date == '2020-10-01') %>%
        select(ADM1_EN, pop_CS, pop_ML1, pop_ML2, CS_3, CS_4, CS_5, ML1_3, ML1_4, ML1_5, ML2_3, ML2_4, ML2_5)

# compute percentages
fn <- fn %>%
              mutate(perc_CS_3p = round((CS_3 + CS_4 + CS_5) * 100 / pop_CS, 1),
                     perc_CS_4p = round((CS_4 + CS_5) * 100 / pop_CS, 1),
                     perc_ML1_3p = round((ML1_3 + ML1_4 + ML1_5) * 100 / pop_ML1, 1),
                     perc_inc_ML1_3p = perc_ML1_3p - perc_CS_3p,
                     period_ML1 = "Oct - Jan 2021")
        

# plot food insecurity current and projected 

li = c(0, 100)
br = c(0, 20, 40, 60, 80, 100)

fi_cs_plot <- eth_adm1[, c('ADM1_EN', 'geometry')] %>% 
                 left_join(fn, by = c('ADM1_EN' = 'ADM1_EN')) %>%
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
                       panel.grid.major = element_blank())


fi_ml1_plot <- eth_adm1[, c('ADM1_EN', 'geometry')] %>% 
                left_join(fn, by = c('ADM1_EN' = 'ADM1_EN')) %>%
                ggplot() +
                geom_sf(aes(fill = perc_ML1_3p), show.legend = T) +   
                geom_sf_label(aes(label = ADM1_EN), color = "black") +
                scale_fill_gradient(low = "white", 
                                    high = "red", 
                                    na.value = NA, 
                                    "Percentage of \n regional population",
                                    breaks=br,
                                    limits=li) +
                ggtitle('Projected by Jan 2021') +
                        #, subtitle = "Crisis or worse levels as reported by FewsNet") +
                theme(axis.text.x=element_blank(),
                      axis.ticks.x=element_blank(),
                      axis.text.y=element_blank(),
                      axis.ticks.y=element_blank(),
                      panel.background = element_blank(),
                      panel.grid.major = element_blank())


fi_inc_ml1_plot <- eth_adm1[, c('ADM1_EN', 'geometry')] %>% 
                    left_join(fn, by = c('ADM1_EN' = 'ADM1_EN')) %>%
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

layout <- fi_cs_plot | fi_ml1_plot 
layout +
   plot_annotation(
    title = 'Food Insecurity',
    subtitle = 'Crisis (IPC 3) or worse levels',
    caption = 'Data source: FewsNet'
  )
