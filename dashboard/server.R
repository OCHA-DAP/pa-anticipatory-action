#
# This is the server logic of the dashboard.
# About deploying to ShinyApps.io: You should not have an explicit install.packages() call within your ui.R or server.R files.

# load libraries
library(DT)

# source code generating all maps
source("generate_subnational_map.R")

# Define server logic
server <- function(input, output) {
    
    # display map of food insecurity triggered regions
    output$trigger_map <- renderPlot({
        trigger_map
    })
    
    # Generate a summary of the data ----
    output$projections_table<- DT::renderDataTable({
        projections_table <- ipc_indices_data_latest %>%
            select(source, ADMIN1, perc_CS_3p, perc_CS_4, perc_ML1_3p, perc_ML1_4, perc_ML2_3p, perc_ML2_4) %>%
            rename(Current_Situation_IPC3plus = perc_CS_3p,
                   Current_Situation_IPC4plus = perc_CS_4,
                   Proj_IPC3plus = perc_ML2_3p,
                   Proj_IPC4plus = perc_ML2_4) %>%
            mutate(Current_Situation_IPC3plus = round(Current_Situation_IPC3plus, 1),
                   Current_Situation_IPC4plus = round(Current_Situation_IPC4plus, 1),
                   Proj_IPC3plus = round(Long_term_proj_IPC3plus, 1),
                   Proj_IPC4plus = round(Long_term_proj_IPC4plus, 1))
        
        projections_table_searchable <- datatable(projections_table,
                                                  filter = list(position = 'top', clear = FALSE),
                                                  options = list(
                                                        columnDefs = list(list(className = 'dt-center', targets = "_all")),
                                                        search = list(regex = TRUE, caseInsensitive = TRUE),
                                                        pageLength = 20))
        
        projections_table_searchable
    })
    
    # Display rainfall maps as temp projections. NOte single quotes must be outer set because string must include double quotes to be processed as HTML
    output$iri <- renderText({c('<img src="','https://iri.columbia.edu/climate/forecast/net_asmt_nmme/2021/feb2021/images/MAM21_Afr_pcp.gif"','width = "500px" height = "500px"', '>')})
    output$icpac <- renderText({c('<img src="','https://www.icpac.net/media/images/MAM_GHA_Rainfall.height-600.width-600.png"','width = "500px" height = "500px"', '>')})
    output$chc <- renderText({c('<img src="','https://blog.chc.ucsb.edu/wp-content/uploads/2021/01/Screen-Shot-2021-01-20-at-5.21.51-PM.png"','width = "500px" height = "500px"', '>')})
    output$nmme <- renderText({c('<img src="','https://www.cpc.ncep.noaa.gov/products/international/nmme/probabilistic_seasonal/africa_nmme_prec_3catprb_FebIC_Mar2021-May2021.png"','width = "600px" height = "500px"', '>')})
    #output$eth_nma <- renderText("No forecast for the 2021 Belg season available from the NMA.")
    output$copernicus <- renderText({c('<img src="','https://apps.ecmwf.int/webapps/opencharts/streaming/20210216-0810/87/pdf2svg-worker-commands-69c6db9bf8-d9qk2-6fe5cac1a363ec1525f54343b6cc9fd8-6Xzy4c.svg"','width = "500px" height = "500px"', '>')})    
    
    # create list of food security triggered regions
    output$triggered_regions_list <- renderText({
                  triggered_regions_list <- trigger_list %>% filter(threshold_reached_H1_2021 == 1) %>% data.frame()
                  
                  return(triggered_regions_list$ADM1_EN)
       })
   
    # create text variable of selected period
   output$period <- renderText(input$period)
}

