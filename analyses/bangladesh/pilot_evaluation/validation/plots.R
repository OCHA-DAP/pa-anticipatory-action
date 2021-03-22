# Faceted line plots showing comparison between Sentinel flooding and Gaussian flooding
# Faceted by mauza with data filered according to a given condition
# If map == TRUE, will also return a map highlighting the mauzas that are selected.
# If cnd == 'rand', then 56 random mauzas will be selected to be mapped. 
gaussian_qa <- function(cnd, df_ts_intp, df_ts_sent, df_flooding, map){
  
  if(cnd=='rand'){
    sel <- df_flooding[sample(nrow(df_flooding), 16),] %>%
      select(PCODE)
  }
  else{
    sel <- df_flooding %>%
      filter(cnd)%>%
      select(PCODE)
  }
  
  ts_intp <- df_ts_intp %>%
    filter(PCODE %in% sel$PCODE) %>%
    mutate(DATE = as.Date(DATE, format = '%Y-%m-%d'))%>%
    mutate(type = 'Gaussian') %>%
    rename(date = DATE, MAUZ_PCODE = PCODE, flooded_fraction = FLOOD_EXTENT)
  
  ts_sent <- df_ts_sent %>%
    filter(MAUZ_PCODE %in% sel$PCODE)%>%
    mutate(date = as.Date(date, format = '%Y-%m-%d'))%>%
    mutate(type = 'Satellite')
  
  ts = rbind(ts_intp, ts_sent)
  
  g <- ggplot(data=ts, aes(date, flooded_fraction, color=type))+
    geom_line()+
    facet_wrap(~MAUZ_PCODE, ncol=8)+
    theme_bw()+
    theme(strip.text.x = element_blank())+
    labs(y='Flooded fraction', x='Date')+
    theme(legend.position="bottom")
  
  if(map == TRUE){
    
    shp_sel <- shp %>%
      mutate(HLT = case_when(OBJECTID %in% sel$PCODE ~ 1, TRUE ~ 0))
    
    m <- tm_shape(shp_sel) +
      tm_layout(frame =FALSE) +
      tm_fill(col='HLT', palette='OrRd', legend.show = FALSE)
    return(list(g, m))
  }
  
  return(g)
}

# Create a choropleth map of a given flooding variable
choro_map <- function(df_flooding, shp, var, pal, hist, mode){
  
  tmap_mode(mode)
  
  df_flooding <- df_summ %>%
    mutate(PEAK_SAT = as.Date(PEAK_SAT, format = '%Y-%m-%d')) %>%
    mutate(PEAK_G = as.Date(PEAK_G, format = '%Y-%m-%d')) %>%
    mutate(PEAK_SAT_DAYS = as.integer(PEAK_SAT -
                                        as.Date('2020-06-01', format = '%Y-%m-%d')))%>%
    mutate(PEAK_G_DAYS = as.integer(PEAK_G -
                                      as.Date('2020-06-01', format = '%Y-%m-%d')))%>%
    mutate(MAX_DIFF = MAX_G - MAX_SAT)%>%
    mutate(FWHM = ifelse(FWHM < 200 & FWHM > 0, FWHM, NA))%>%
    mutate(COV = ifelse(COV < 50, FWHM, NA))
  
  shp_flooding <- shp %>%
    select(OBJECTID, geometry)%>%
    right_join(df_flooding, by=c('OBJECTID'='PCODE'))
  
  
  m <- tm_shape(shp_flooding) +
    tm_layout(frame =FALSE,
              legend.outside = T, 
              legend.outside.position = 'left',) +
    tm_fill(col=var,
            colorNA = '#eb4034',
            palette=pal,
            style='fisher', 
            title='',
            legend.hist = hist)
  return(m)
  
}

# Faceted plot to compare between GloFAS, Gaussian, and Sentinel 
# for overall dynamics of flooding 
compare_glofas <- function(df_sent, df_gaus, df_glofas){
  
  station_mauz <- st_join(shp_mauz, stations)%>%
    drop_na(Station) %>%
    select(c(OBJECTID, Station, geometry))
  
  glofas <- df_glofas %>%
    gather(station, discharge, dis24_Noonkhawa:dis24_Aricha)%>%
    mutate(station = substr(station, 7,nchar(station)))
  
  joined <- glofas %>%
    left_join(station_mauz, by=c('station'='Station'))%>%
    left_join(df_sent, by=c('OBJECTID'='MAUZ_PCODE', 'date'='date'))%>%
    left_join(df_gaus, by=c('OBJECTID'='PCODE', 'date'='DATE'))%>%
    drop_na(OBJECTID)%>%
    filter(date>'2020-06-01' & date<'2020-09-01')%>%
    rename(Gaussian = FLOOD_EXTENT)%>%
    rename(Sentinel = flooded_fraction) %>%
    rename(GloFAS = discharge) %>%
    mutate(date = as.Date(date))%>%
    gather(measure, value, c(GloFAS, Sentinel, Gaussian))%>%
    select(date, station, OBJECTID, measure, value)
  
  g <- ggplot(data=joined[!is.na(joined$value),], aes(x=date))+
    geom_line(aes(y=value), na.rm=TRUE)+
    facet_grid(rows=vars(measure), cols=vars(station), scales='free_y')+
    theme_bw()+
    labs(y='Flooding value', x='Date')
  
  return(g)
  
  
}

# Simple maps to show the study area
study_area <- function(adm2, adm0, shp_river, mode){
  
  tmap_mode(mode)
  
  adm2 <- adm2%>%
    filter(ADM2_EN %in% c('Bogra', 'Gaibandha', 'Jamalpur', 'Kurigram', 'Sirajganj'))
  
  map <- tm_basemap(leaflet::providers$Esri.WorldTopoMap) +
    #tm_shape(adm0)+
    #tm_polygons(col='#d1d1d1', alpha=0.5)+
    tm_shape(adm2)+
    tm_polygons(col='#fc6f6f', alpha=0.3, id='ADM2_EN')+
    tm_shape(shp_river)+
    tm_fill(col="#010885", id='')+
    tm_layout(frame=FALSE)+
    tm_shape(stations)+
    tm_dots(col='#ff6830')
  return(map)
}

