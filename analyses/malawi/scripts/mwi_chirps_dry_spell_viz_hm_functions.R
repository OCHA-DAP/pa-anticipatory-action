# Transform the data for plotting -----------------------------------------
fill_dates <- function(df_dates, df_rainy_season,start_col, end_col, fill_num){
  adms <- unique(df_rainy_season[['pcode']])
  dates <- seq(as.Date("2000-01-01"), as.Date("2020-12-31"), by="days")
  df <- data.frame(dates)
  for (adm in adms){
    df[adm] <- 0
    for(i in 1:nrow(df_dates)){
      if (df_dates$pcode[i] == adm){
        ds_start <- df_dates[[start_col]][i]
        ds_end <- df_dates[[end_col]][i]
        df[[adm]][(df$date > ds_start) & (df$date <= ds_end)] <- fill_num
      }
    }
  }
  # Clean up the df to a tidy format for ggplot
  df_long <- df %>%
    gather(pcode, day_type, head(adms,1):tail(adms,1))%>%
    mutate(day_fac = as.factor(day_type))%>%
    mutate(month_day = format(dates, "%m-%d"))%>%
    mutate(date_no_year = as.Date(paste0('1800-',month_day), "%Y-%m-%d"))%>%
    mutate(year = lubridate::year(dates))%>%
    mutate(region = substr(pcode, 3, 3)) %>%
    mutate(region = ifelse(region == 3, "Southern", ifelse(region == 2, "Central", "Northern")))
  
  return(df_long)
}

#part of function fill_dates, used if data already in semi-processed format
prepare_ggplot <- function(df,ds_col){
  df$dates <- as.Date(df$date,format="%Y-%m-%d")
  df$day_type <- df[[ds_col]]
  df_long <- df %>%
    mutate(day_fac = as.factor(day_type))%>%
    mutate(month_day = format(dates, "%m-%d"))%>%
    mutate(date_no_year = as.Date(paste0('1800-',month_day), "%Y-%m-%d"))%>%
    mutate(year = lubridate::year(dates)) %>%
    mutate(region = substr(pcode, 3, 3)) %>%
    mutate(region = ifelse(region == 3, "Southern", ifelse(region == 2, "Central", "Northern")))
  return(df_long)
}

plot_heatmap <- function(df_dry_spells,df_rainy_season, match_values,match_labels,color_scale,y_label,plot_title,output_path,ds_flatdata=FALSE,sub_title=""){
  #ds_flatdata: choose the appropriate preprocessing, depending on the format the data comes in
  #the output from mwi_chirps_dry_spell_detection.R should use ds_flatdata=FALSE
  #however, some cases the data is preprocessed in python in which case ds_flatdata=TRUE
  if (ds_flatdata) {
    df_ds <- prepare_ggplot(df_dry_spells,"dryspell_match")
  }
  else {
    df_ds <- fill_dates(df_dry_spells, df_rainy_season,'dry_spell_first_date', 'dry_spell_last_date', 1)
  }
  df_rs <- fill_dates(df_rainy_season, df_rainy_season,'onset_date', 'cessation_date', 10)
  
  theme_hm <- theme_minimal()+
    theme(
      axis.text.y=element_blank(),
      axis.ticks.y=element_blank(),
      legend.position = 'bottom',
      strip.text = element_text(size=16,angle=0),
      axis.text.x = element_text(size=16),
      legend.text = element_text(size=16),
      axis.title.x = element_text(size=16),
      axis.title.y = element_text(size=16),
      plot.title = element_text(size=32)
    )
  
  
  hm_plot <- df_ds %>%
    full_join(df_rs, by=c('pcode', 'dates'))%>%
    mutate(days = as.factor(day_type.x + day_type.y))%>%
    mutate(days = factor(days, levels=dry_spell_match_values, labels=match_values_labels))%>%
    drop_na(date_no_year.x) %>%
    arrange(desc(pcode),date_no_year.x) %>%
    ggplot(aes(x=date_no_year.x, y=pcode, fill=days))+
    geom_tile() +
    scale_fill_manual(values=color_scale)+
    facet_grid(rows=vars(year.x))+
    theme_minimal()+
    labs(title=plot_title, subtitle=sub_title,x='Date', y=y_label, fill='')+
    theme_hm+
    scale_x_date(date_labels = "%b",date_breaks = "1 month",expand=c(0,0))
  return(hm_plot)
}