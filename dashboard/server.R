#
# This is the server logic of the dashboard.
# About deploying to ShinyApps.io: You should not have an explicit install.packages() call within your ui.R or server.R files.

# load libraries
library(DT)

# source code generating all maps
source("generate_subnational_map.R")

# Define server logic
server <- function(input, output) {
    
    # create variables for period values
    fn_ml1 <- ipc_indices_data %>% filter(Source == 'FewsNet') %>% select(ML1_period) %>% unique() %>% as.character()
    fn_ml2 <- ipc_indices_data %>% filter(Source == 'FewsNet') %>% select(ML2_period) %>% unique() %>% as.character()
    gbl_ml1 <- ipc_indices_data %>% filter(Source == 'GlobalIPC') %>% select(ML1_period) %>% unique() %>% as.character()
    gbl_ml2 <- ipc_indices_data %>% filter(Source == 'GlobalIPC') %>% select(ML2_period) %>% unique() %>% as.character()

    # conditionally select correct map
    output$trigger_map <- renderPlot({
        if (input$country == 'eth' & input$source == 'fn' & input$period == fn_ml1)
            return(eth_fn_ML1_trigger_map)
        
        if (input$country == 'eth' & input$source == 'fn' & input$period == fn_ml2)
            return(eth_fn_ML2_trigger_map)
        
        if (input$country == 'eth' & input$source == 'gbl' & input$period == gbl_ml1)
            return(eth_gbl_ML1_trigger_map)
        
        if (input$country == 'eth' & input$source == 'gbl' & input$period == gbl_ml2)
            return(eth_gbl_ML2_trigger_map)
    })
    
    # dynamically create options for period radio buttons
    output$projectionPeriods <- renderUI({
        source <- switch(input$source,
                       fn = 'FewsNet',
                       gbl = 'GlobalIPC') 
        
        period_options <- ipc_indices_data %>%
                             filter(Source == source) %>% 
                             select(ML1_period, ML2_period) %>%
                             unique() %>% 
                             as.character()
        
        radioButtons('period', 'Select a projection period:', 
                     period_options)
                    # selected = character(0)) # no default
    })
    
    # Generate a summary of the data ----
    output$projections_table<- DT::renderDataTable({
        projections_table <- ipc_indices_data %>%
            select(Source, ADMIN1, perc_CS_3p, perc_CS_4, perc_ML1_3p, perc_ML1_4, perc_ML2_3p, perc_ML2_4) %>%
            rename(Current_Situation_IPC3plus = perc_CS_3p,
                   Current_Situation_IPC4plus = perc_CS_4,
                   Short_term_proj_IPC3plus = perc_ML1_3p,
                   Short_term_proj_IPC4plus = perc_ML1_4,
                   Long_term_proj_IPC3plus = perc_ML2_3p,
                   Long_term_proj_IPC4plus = perc_ML2_4) %>%
            mutate(Current_Situation_IPC3plus = round(Current_Situation_IPC3plus, 1),
                   Current_Situation_IPC4plus = round(Current_Situation_IPC4plus, 1),
                   Short_term_proj_IPC3plus = round(Short_term_proj_IPC3plus, 1),
                   Short_term_proj_IPC4plus = round(Short_term_proj_IPC4plus, 1),
                   Long_term_proj_IPC3plus = round(Long_term_proj_IPC3plus, 1),
                   Long_term_proj_IPC4plus = round(Long_term_proj_IPC4plus, 1))
        
        projections_table_searchable <- datatable(projections_table,
                                                  filter = list(position = 'top', clear = FALSE),
                                                  options = list(
                                                        columnDefs = list(list(className = 'dt-center', targets = "_all")),
                                                        search = list(regex = TRUE, caseInsensitive = TRUE),
                                                        pageLength = 20))
        
        projections_table_searchable
    })
    
    # Display rainfall maps as temp projections. NOte single quotes must be outer set because string must include double quotes to be processed as HTML
    
    output$iri <- renderText({c('<img src="','https://iri.columbia.edu/climate/forecast/net_asmt_nmme/2020/nov2020/images/MAM21_Afr_pcp.gif"','width = "500px" height = "500px"', '>')})
    output$icpac <- renderText({c('<img src="','https://www.icpac.net/media/images/Rainfall_forecast_november-January_.height-600.width-600.png"','width = "500px" height = "500px"', '>')})
    output$chc <- renderText({c('<img src="','https://blog.chc.ucsb.edu/wp-content/uploads/2020/11/Screen-Shot-2020-11-05-at-12.39.37-PM.png"','width = "500px" height = "500px"', '>')})
    output$nmme <- renderText({c('<img src="','https://www.cpc.ncep.noaa.gov/products/NMME/prob/images/prob_ensemble_prate_season4.png"','width = "500px" height = "500px"', '>')})
    output$eth_nma <- renderText("No forecast for the 2021 Belg season available from the NMA.")
        
    # create conditional lists of triggered regions
    output$triggered_regions_list <- renderText({
       
       if(input$country == 'eth' & input$source == 'fn' & input$period == fn_ml1){
          triggered_regions_list <- eth_fn_ML1_trigger_list$ADM1_EN
       } 
        
       if(input$country == 'eth' & input$source == 'fn' & input$period == fn_ml2){
          triggered_regions_list <- eth_fn_ML2_trigger_list$ADM1_EN
       }
        
       if(input$country == 'eth' & input$source == 'gbl' & input$period == gbl_ml1){
        #  triggered_regions_list <- ifelse(length(eth_gbl_ML1_trigger_list$ADM1_EN) > 0, eth_gbl_ML1_trigger_list$ADM1_EN, "No region meets the trigger")
           triggered_regions_list <- eth_gbl_ML1_trigger_list$ADM1_EN
       }
        
       if(input$country == 'eth' & input$source == 'gbl' & input$period == gbl_ml2){
           triggered_regions_list <- eth_gbl_ML2_trigger_list$ADM1_EN
       }
        triggered_regions_list
        
       })
   
    # create text variable of selected period
   output$period <- renderText(input$period)
}

