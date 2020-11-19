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
    output$projections_table<- renderDataTable({
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
                                                        pageLength = 11))
        
        projections_table_searchable
    })
    
    # Publish clickable links to full reports
    
    fewsnet_url <- a("FewsNet", href = "https://fews.net/east-africa/ethiopia/food-security-outlook/october-2020")
    globalipc_url <- a("GlobalIPC", href = "http://www.ipcinfo.org/ipc-country-analysis/details-map/en/c/1152818/?iso3=ETH")
              
    output$reports <- renderUI({
            tagList("Full reports:", 
                    fewsnet_url,
                   globalipc_url)
        })
    
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

