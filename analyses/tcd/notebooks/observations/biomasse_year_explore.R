# just trying to map out years of Biomasse to see if 2017 was localized

###############
#### SETUP ####
###############

library(tidyverse)
library(sf)

aa_dir <- Sys.getenv("AA_DATA_DIR")

df_priority <- readxl::read_excel(file.path(
  aa_dir, "private", "exploration", "tcd", "drought", "tcd_priority_adm2.xlsx"
),
sheet = "Sheet1") %>%
  transmute(Departement = case_when(
              Departement == "Sud Kanem" ~ "Wadi-Bissam",
              Departement == "IRIBA" ~ "Kobé",
              Departement == "Dar Tama" ~ "Dar-Tama",
              TRUE ~ str_replace(Departement, "El-Gazal", "El-Gazel")
            ),
            area_of_interest = TRUE)

df <- read_sf(file.path(
  aa_dir, "public", "raw", "glb", "biomasse", "WA_DMP_ADM2_ef_v0.csv"
)) %>%
  filter(admin0Name == "Chad") %>%
  transmute(admin2Name,
            the_geom,
            across(matches("BIO_[0-9]{4}"), ~ 100 * as.numeric(.x) / as.numeric(BIO_MEAN))) %>%
  pivot_longer(starts_with("BIO"),
               names_to = "year",
               names_prefix = "BIO_") %>%
  filter(between(year, 1999, 2021)) %>%
  left_join(df_priority, by = c("admin2Name" = "Departement")) %>%
  mutate(the_geom = st_as_sfc(the_geom)) %>%
  st_as_sf()

#################
#### MAPPING ####
#################

df %>%
  filter(area_of_interest) %>%
  st_as_sf() %>%
  ggplot() +
  geom_rect(data = data.frame(year = c(2002, 2004, 2006,  2009, 2011)),
            fill = "lightgrey",
            alpha = 0.3,
            color = "darkgrey",
            ymin = -Inf,
            ymax = Inf,
            xmin = -Inf,
            xmax = Inf) +
  geom_text(data = data.frame(year = 2022),
            label = str_wrap("2017 not localized drought, just less severe compared to other years", 20),
            x = 17,
            y = 15,
            size = 3,
            fontface = "bold") +
  geom_sf(aes(fill = value)) +
  facet_wrap(~year, labeller = as_labeller(\(x, y) c(1999:2021, ""))) +
  theme_void() +
  scale_fill_steps2(midpoint = 100,
                    breaks = c(0, 50, 80, 90, 100, 250)) +
  labs(title = "Biomasse anomaly, Chad ADM2 levels",
       subtitle = "5 worst years highlighted")

# look at missed years

df %>%
  filter(area_of_interest) %>%
  st_as_sf() %>%
  ggplot() +
  geom_rect(data = data.frame(year = c(2001, 2017)),
            fill = "#EEEEEE",
            alpha = 1,
            color = "red",
            lwd = 1.5,
            ymin = -Inf,
            ymax = Inf,
            xmin = -Inf,
            xmax = Inf) +
  geom_sf(aes(fill = value)) +
  facet_wrap(~year, labeller = as_labeller(\(x, y) c(1999:2021, ""))) +
  theme_void() +
  scale_fill_steps2(midpoint = 100,
                    breaks = c(0, 50, 80, 90, 100, 250)) +
  labs(title = "Biomasse anomaly, Chad ADM2 levels",
       subtitle = "2 missed activations highlighted")
