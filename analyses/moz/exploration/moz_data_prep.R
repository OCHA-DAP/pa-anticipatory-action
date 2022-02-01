library(tidyverse)
library(gt)
library(sf)
library(rmapshaper)
library(fuzzyjoin)

#################
#### LOADING ####
#################

file_dir <- file.path(
  Sys.getenv("AA_DATA_DIR"),
  "private",
  "raw",
  "moz",
  "AA data Nampula Cabo delgado"
)

df_list <- map(
  list.files(file_dir),
  ~ readxl::read_excel(file.path(file_dir, .x), skip = 1)
)

df <- reduce(
  df_list,
  full_join,
  by = "organisationunitname"
)

sf_dir <- file.path(
  Sys.getenv("AA_DATA_DIR"),
  "public",
  "raw",
  "moz",
  "cod_ab"
)

adm1_sf <- read_sf(
  file.path(
    sf_dir,
    "moz_admbnda_adm1_ine_20190607.shp"
  )
)

adm2_sf <- read_sf(
  file.path(
    sf_dir,
    "moz_admbnda_adm2_ine_20190607.shp"
  )
)

adm3_sf <- read_sf(
  file.path(
    sf_dir,
    "moz_admbnda_adm3_ine_20190607.shp"
  )
)

###################
#### WRANGLING ####
###################


long_df <- df %>%
  pivot_longer(
    -organisationunitname,
    names_to = c("date", "epi_type"),
    names_sep = " - ",
    values_to = "cases"
  ) %>%
  mutate(
    year = as.numeric(str_extract(date, "[0-9]{4}")),
    epiweek = as.numeric(str_extract(date, "(?<=W)[0-9]+")),
    date = paste(year, epiweek, sep = ", "),
    year_week = (51 * (year - 2017)) + epiweek,
    organisationunitname = case_when( # repairing names for later joining to geodataframe as necessary
      organisationunitname == "CHIÚRE" ~ "Chiure",
      organisationunitname == "MOCÍMBOA DA PRAIA" ~ "Mocimboa Da Praia",
      organisationunitname == "DISTRITO DE NAMPULA" ~ "Cidade De Nampula",
      organisationunitname == "LIUPO" ~ "Liúpo",
      organisationunitname == "NACALA-PORTO" ~ "Nacala",
      TRUE ~ str_to_title(organisationunitname)
    ),
    epi_type = case_when( # changing names for easy manipulation
      epi_type == "CÓLERA CASOS" ~ "cholera_all",
      epi_type == "DIARREIA 0-4 anos, CASOS" ~ "diarrhea_04",
      epi_type == "DIARREIA 15+ anos, CASOS" ~ "diarrhea_15plus",
      epi_type == "DIARREIA 5-14 anos, CASOS" ~ "diarrhea_514",
      epi_type == "DISENTERIA CASOS" ~ "dysentery_all"
    ),
    cases = replace_na(cases, 0)
  ) %>%
  pivot_wider(names_from = "epi_type", values_from = "cases") %>%
  rowwise() %>%
  mutate(diarrhea_all = sum(c_across(starts_with("diarrhea")))) %>%
  pivot_longer(-c(organisationunitname, date, year, epiweek, year_week),
               names_to = c("epi_type", "age_group"),
               names_sep = "_",
               values_to = "cases") %>%
  mutate(
    age_group = case_when(
      age_group == "04" ~ "0 - 4",
      age_group == "514" ~ "5 - 14",
      age_group == "15plus" ~ "15+",
      TRUE ~ age_group
    )
  ) %>%
  bind_rows(.,
            filter(., organisationunitname %in% c("Cabo Delgado", "Nampula")) %>%
              group_by(date, year, epiweek, year_week, epi_type, age_group) %>%
              summarize(cases = sum(cases),
                        organisationunitname = "Mozambique",
                        .groups = "drop")
  ) %>%
  group_by(organisationunitname, epi_type, age_group) %>%
  arrange(year_week, .by_group = TRUE) %>%
  mutate(
    cases_2_weeks = zoo::rollsum(cases, 2, na.pad = T, align = "right"),
    cases_3_weeks = zoo::rollsum(cases, 3, na.pad = T, align = "right"),
    cases_4_weeks = zoo::rollsum(cases, 4, na.pad = T, align = "right")
  ) %>%
  ungroup()

# separate out aggregated province data from district data
district_df <- long_df %>%
  mutate(
    province = ifelse(
      organisationunitname %in% c("Cabo Delgado", "Nampula"),
      organisationunitname,
      NA)
  ) %>%
  fill(province) %>%
  filter(!(organisationunitname %in% c("Nampula", "Cabo Delgado", "Mozambique"))) %>%
  rename(district = organisationunitname)

province_df <- filter(long_df, organisationunitname %in% c("Nampula", "Cabo Delgado")) %>%
  rename(province = organisationunitname)

overall_df <- filter(long_df, organisationunitname == "Mozambique") %>%
  select(-organisationunitname)

# simplify the polygons because plotting is taking up to 10 - 15 minutes
adm1_sf <- rmapshaper::ms_simplify(input = as(adm1_sf, "Spatial")) %>%
  st_as_sf()

adm2_sf <- rmapshaper::ms_simplify(input = as(adm2_sf, "Spatial")) %>%
  st_as_sf()

adm3_sf <- rmapshaper::ms_simplify(input = as(adm3_sf, "Spatial")) %>%
  st_as_sf()

adm1_cdn_sf <- adm1_sf %>%
  filter(ADM1_PCODE %in% c("MZ01", "MZ07")) %>%
  st_as_sf()