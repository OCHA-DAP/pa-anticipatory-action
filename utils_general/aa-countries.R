###
# Install packages
###
library(tidyverse)

###
# set up paths
###
data_dir <- Sys.getenv("AA_DATA_DIR")
plot_path <- paste0(data_dir, "/public/processed/glb/plots")

###
# get world map
###
world_map <- map_data("world")

###
# identify AA pilots
###

# list countries and shocks
drought <- c("Somalia", "Ethiopia", "Niger", "Burkina Faso")
floods <- c("Bangladesh", "Nepal", "Chad", "South Sudan")
dual <- c("Malawi")
storms <- c("Philippines") # There is no "The" in the world_map
infectious <- c("Madagascar")

aa_countries_list <- data.frame(country = drought, shock = "Drought") 
aa_countries_list <- rbind(aa_countries_list, data.frame(country = dual, shock = "Dry Spells + Floods"))
aa_countries_list <- rbind(aa_countries_list, data.frame(country = floods, shock = "Floods"))
aa_countries_list <- rbind(aa_countries_list, data.frame(country = storms, shock = "Storms"))
aa_countries_list <- rbind(aa_countries_list, data.frame(country = infectious, shock = "Infectious"))

length(unique(aa_countries_list$country)) # number of unique countries

world_map$aa_country <- ifelse(world_map$region %in% aa_countries_list$country, 1, 0) # binary is a AA country or not
world_map$aa_country <- as.factor(world_map$aa_country)
world_map %>% filter(aa_country == 1) %>% summarise(unique(region)) # check that every aa country is in world map with same spelling as above

world_map <- world_map %>% # add shock info to map info
              left_join(aa_countries_list, by = c('region' = 'country'))

world_map$shock <- as.factor(world_map$shock) # formatting

aa_countries_map <-  world_map %>% filter(!is.na(shock)) # subset mapping + shock info for aa countries
  
###
# map AA countries
###

# select colors for each shock, in order: drought, dry spells + floods, floods, infectious, storms
colours <- gplots::col2hex(c("lightgoldenrod2", "springgreen4" , "dodgerblue3", "chocolate2", "lightskyblue"))

# plot map and colorise AA countries
ggplot(world_map, aes(x = long, y = lat, group = group)) +
  geom_polygon(fill = "#CCCCCC") +
  geom_polygon(data = aa_countries_map, aes(x = long, y = lat, group = group, fill = shock), show.legend = TRUE) +
  scale_fill_manual(values =  colours, name = "Shocks") +
  labs(title = 'Countries with Anticipatory Action Frameworks'
       ,subtitle = "Active and under development frameworks") +
  theme(legend.position="bottom"
        ,plot.title = element_text(size = 24)
        ,plot.subtitle = element_text(size = 12)
        ,axis.text = element_blank()
        ,axis.title = element_blank()
        ,axis.ticks = element_blank()
        ,panel.background = element_rect(fill = "white")
        ,panel.grid = element_blank()
  )

# save plot ## NOTE Programmatically saving the image distorts it. Better save through the GUI for now until we figure out the issue.
ggsave(paste0(plot_path, "/aa_countries_map.png"), width = 1426, height = 744, units = "px", dpi = 300)

