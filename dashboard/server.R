
# load libraries
library(DT)

# source code generating all maps
source("generate_subnational_map.R")

# Define server logic
server <- function(input, output) {
    
    # Display  last update date
    output$last_update <- renderText({paste0("Analysis last updated: ", "16 Feb 2021")})
    
    url <-  a("Detailed Summary. ", href="https://docs.google.com/document/d/1xa8AmYVpw1z7moYwFrZoQ4S2o5g-QbC8U-G-3Zlpgjw/edit?usp=sharing")
    output$summary_link <- renderUI({
        tagList("Read the full", url)
    })
    
    
    # display map of food insecurity triggered regions
    output$trigger_map <- renderPlot({
        fs_trigger_map
    })
    
    # Generate a summary of the data ----
    output$projections_table <- DT::renderDataTable({
                                        
                                        projections_table_fn <- ipc_indices_data_latest %>%
                                            mutate(threshold_reached_H1_2021 = ifelse((source == 'FewsNet' & threshold_reached_ML2 == 'True') | (source == 'GlobalIPC' & threshold_reached_ML1 == 'True'), "Yes", "No")) %>%
                                            filter(source == 'FewsNet') %>%
                                            rename(Oct2020_IPC3plus = perc_CS_3p,
                                                   Proj_IPC3plus = perc_ML2_3p,
                                                   Proj_IPC4plus = perc_ML2_4,
                                                   Trigger_met = threshold_reached_H1_2021) %>%
                                            mutate(Oct2020_IPC3plus = round(Oct2020_IPC3plus, 1),
                                                   Proj_IPC3plus = round(Proj_IPC3plus, 1),
                                                   Change_IPC3plus = round(Proj_IPC3plus - Oct2020_IPC3plus, 1),
                                                   Proj_IPC4plus = round(Proj_IPC4plus, 1)) %>%
                                            select(source, ADMIN1, Oct2020_IPC3plus, Proj_IPC3plus,  Change_IPC3plus, Proj_IPC4plus, Trigger_met)
                                        
                                        projections_table_gb <- ipc_indices_data_latest %>%
                                            mutate(threshold_reached_H1_2021 = ifelse((source == 'FewsNet' & threshold_reached_ML2 == 'True') | (source == 'GlobalIPC' & threshold_reached_ML1 == 'True'), "Yes", "No")) %>%
                                            filter(source == 'GlobalIPC') %>%
                                            rename(Oct2020_IPC3plus = perc_CS_3p,
                                                   Proj_IPC3plus = perc_ML1_3p,
                                                   Proj_IPC4plus = perc_ML1_4, 
                                                   Trigger_met = threshold_reached_H1_2021) %>%
                                            mutate(Oct2020_IPC3plus = round(Oct2020_IPC3plus, 1),
                                                   Proj_IPC3plus = round(Proj_IPC3plus, 1),
                                                   Change_IPC3plus = round(Proj_IPC3plus - Oct2020_IPC3plus, 1),
                                                   Proj_IPC4plus = round(Proj_IPC4plus, 1)) %>%
                                            select(source, ADMIN1, Oct2020_IPC3plus, Proj_IPC3plus,  Change_IPC3plus, Proj_IPC4plus, Trigger_met)

                                        projections_table <- rbind(projections_table_fn, projections_table_gb)
                                        
        projections_table_searchable <- datatable(projections_table,
                                                  filter = list(position = 'top', clear = FALSE),
                                                  options = list(
                                                        columnDefs = list(list(className = 'dt-center', targets = "_all")),
                                                      #  search = list(regex = TRUE, caseInsensitive = TRUE),
                                                        pageLength = 20))
                                                  
        projections_table_searchable                                     
    })
    
    # Display rainfall maps as temp projections. Note single quotes must be outer set because string must include double quotes to be processed as HTML
    output$iri_text <- renderText("Below: IRI for AMJ season")
    output$iri <- renderText({c('<img src="','https://iri.columbia.edu/climate/forecast/net_asmt_nmme/2021/mar2021/images/AMJ21_Afr_pcp.gif"','width = "500px" height = "500px"', '>')})
    output$nmme_text <- renderText("Below: NMME for AMJ season")
    output$nmme <- renderText({c('<img src="','https://www.cpc.ncep.noaa.gov/products/international/nmme/probabilistic_seasonal/africa_nmme_prec_3catprb_MarIC_Apr2021-Jun2021.png"','width = "600px" height = "500px"', '>')})
    output$copernicus_text <- renderText("Below: Copernicus for AMJ season")
    output$copernicus <- renderText({c('<img src="','https://apps.ecmwf.int/webapps/opencharts/streaming/20210316-0730/31/pdf2svg-worker-commands-88596cfc-pk72d-6fe5cac1a363ec1525f54343b6cc9fd8-qIoeO3.svg"','width = "500px" height = "500px"', '>')})    
    output$icpac_text <- renderText("Below: ICPAC for MAM season")
    output$icpac <- renderText({c('<img src="','https://www.icpac.net/media/images/MAM_GHA_Rainfall.height-600.width-600.png"','width = "500px" height = "500px"', '>')})
    output$chc_text <- renderText("Below: Climate Hazards Center for MAM season")
    output$chc <- renderText({c('<img src="','https://blog.chc.ucsb.edu/wp-content/uploads/2021/01/Screen-Shot-2021-01-20-at-5.21.51-PM.png"','width = "500px" height = "500px"', '>')})
    
    output$nma_text <- renderText("Below: NMA Ethiopia for FMAM season")
    
    # create list of food security triggered regions
    output$triggered_regions_list <- renderText({
                  triggered_regions_list <- trigger_list %>% filter(threshold_reached_H1_2021 == 1) %>% data.frame()
                  
                  return(triggered_regions_list$ADM1_EN)
       })
   
    # create text variable of selected period
   output$period <- renderText(input$period)
}

