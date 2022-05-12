library(ranger)
library(tidyverse)
library(ncdf4)

data_dir <- Sys.getenv("AA_DATA_DIR")

###################
#### WRANGLING ####
###################

# load and wrangle SFED flood extent data
# into data frame format with dates as rows
# and flattened matrix as columns

ssd_fs <- ncdf4::nc_open(
  file.path(
    data_dir,
    "private",
    "exploration",
    "ssd",
    "floodscan",
    "ssd_floodscan.nc"
  )
)

sfed <- ncvar_get(ssd_fs, "SFED_AREA")

sfed_r <- map(
  1:dim(sfed)[1],
  \(i) {
    x <- as.numeric(sfed[i,,])
    x[!is.na(x)]
  }
)

# bring SFED data into a single data frame as
# described above. drop days on leap years just
# for simplicity now, can deal with in more rigorous
# manner in the future
sfed_df <- as_tibble(do.call(rbind, sfed_r)) %>%
  rename_with(~ gsub("V", "sfed_", .x)) %>%
  mutate(
    date = lubridate::as_date(
      ssd_fs$dim$time$vals,
      origin = str_extract(ssd_fs$dim$time$units, "[0-9]{4}-[0-9]{2}-[0-9]{2}")
    ),
    year = lubridate::year(date),
    day_of_year = lubridate::yday(date)
  ) %>%
  group_by(year) %>%
  filter(
    !((day_of_year == 60) & (366 %in% day_of_year)),
    year <= 2021
  ) %>%
  mutate(
    ifelse(
      366 %in% day_of_year & day_of_year > 60,
      day_of_year - 1,
      day_of_year
    )
  ) %>%
  ungroup() 

########################
#### CLASSIFICATION ####
########################
 
# for classification, let's just generate the date with the max
# area covered for each year and give those return periods
# then build the dataset for classification on those

sfed_sum <- sfed_df %>%
  rowwise() %>%
  mutate(total_sfed = sum(c_across(starts_with("sfed")))) %>%
  ungroup() %>%
  select(year, date, day_of_year, total_sfed)

sfed_max <- sfed_sum %>%
  filter(day_of_year > 90) %>%
  group_by(year) %>%
  filter(total_sfed == max(total_sfed)) %>%
  ungroup()

#############
#### PCA ####
#############

