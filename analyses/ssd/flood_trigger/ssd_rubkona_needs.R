library(tidyverse)

# Explore the latest FSNMS data that covers Bentiu town
# and surrounding Bentiu IDP sites. This can be used
# to inform the potential selection of activities across sites

file_dir <- file.path(
  Sys.getenv("AA_DATA_DIR"),
  "private",
  "raw",
  "ssd",
  "fsnsms"
)

df_camp <- read_csv(
  file.path(
    file_dir,
    "iom_dtm_fsnms_2021_camp_bentiu_restricted.csv"
  ),
  col_types = cols(.default = "c")
) 

df_urban <- read_csv(
  file.path(
    file_dir,
    "iom_dtm_fsnms_2021_urban_bentiu_restricted.csv"
  ),
  col_types = cols(.default = "c")
)

###################
#### WRANGLING ####
###################

# get data down to key columns
# and bind the two data frames
cols <- c(
  "a_location",
  "a_county",
  "e_water_needsmet",
  "e_water_source",
  "e_defecation",
  "j_hunger_sleephungry",
  "j_hunger_wholeday",
  "i_foodcon_cereals",
  "i_foodcon_grains",
  "i_foodcon_roots",
  "i_foodcon_legumes",
  "i_foodcon_dairy",
  "i_foodcon_meat",
  "i_foodcon_flesh",
  "i_foodcon_organ",
  "i_foodcon_fish",
  "i_foodcon_eggs",
  "i_foodcon_veggies",
  "i_foodcon_oveg",
  "i_foodcon_leaf",
  "i_foodcon_fruits",
  "i_foodcon_ofruits",
  "i_foodcon_oil",
  "i_foodcon_sugar"
)

df <- bind_rows(
  df_urban,
  df_camp
) %>%
  select(any_of(cols)) %>%
  mutate(location = ifelse(
    is.na(a_location),
    a_county,
    a_location
  ),
  .before = 1) %>%
  select(-c(a_location, a_county)) %>%
  type_convert()

# calculate the food consumption score
df <- df %>%
  mutate(
    across(
      contains("foodcon"),
      ~ ifelse(
        is.na(.x),
        0,
        as.numeric(substr(.x, 1, 1))
      )
    ),
    i_foodcon_staples = pmin(i_foodcon_cereals + i_foodcon_grains + i_foodcon_roots, 7),
    i_foodcon_meat_tot = pmin(i_foodcon_meat + i_foodcon_flesh + i_foodcon_organ + i_foodcon_fish + i_foodcon_eggs, 7),
    i_foodcon_fruit_tot = pmin(i_foodcon_fruits + i_foodcon_ofruits, 7),
    i_foodcon_veg_tot = pmin(i_foodcon_veggies + i_foodcon_oveg + i_foodcon_leaf, 7),
    fcs = 2 * i_foodcon_staples + 3 * i_foodcon_legumes + i_foodcon_veg_tot + i_foodcon_fruit_tot + 4 * i_foodcon_meat_tot + 4 * i_foodcon_dairy + 0.5 * i_foodcon_sugar + 0.5 * i_foodcon_oil
  )

#################
#### SUMMARY ####
#################

# some quick checks tell us this data is extremely
# problematic and shouldn't be trusted
# 67 households in the actual IDP camp (not the 
# informal sites) report having eaten NOTHING
# in the past 7 days. Since they aren't dead and
# are happily doing a very lengthy, that's clearly
# false.
df %>%
  filter(location == "Bentiu IDP Camp") %>%
  pull(fcs) %>%
  {sum(. == 0)}

# the histograms are drastically different. Although
# the one of the town is near-normal, the results
# just can't reasonably be used because based on
# reported malnutrition, the situation in the IDP
# camp should be much better than in the informal sites,
# yet this shows a very very good situation in the
# informal sites
df %>%
  filter(location == "Bentiu IDP Camp") %>%
  pull(fcs) %>%
  hist()

df %>%
  filter(location == "Rubkona") %>%
  pull(fcs) %>%
  hist()

# sadly I'm going to pull the plug on the
# analysis because these results really show
# we can't trust this data much at all