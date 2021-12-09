########
## Project: Explore food insecurity in Niger based on Cadre Harmonisé data 2014-2021
########

#####
## setup
#####

# load libraries
library(tidyverse)
library(rhdx)

# set options and paths
options(scipen = 999)

# read in data from HDX
# url <- "https://data.humdata.org/dataset/5123033a-2db1-496c-b381-df804ac30595/resource/c689cb9f-2475-4b7e-8beb-178ed9f6253d/download/cadre_harmonise_caf_ipc.xlsx"
set_rhdx_config(hdx_site = "prod")
data_all <- pull_dataset("5123033a-2db1-496c-b381-df804ac30595") %>%
     get_resource(1) %>%
     read_resource()

# select Niger-specific rows
data <- data_all %>%
  subset(adm0_pcod3 == 'NER')
